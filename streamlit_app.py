import streamlit as st
import re
import os
import json
import time
import shutil
import atexit
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dateutil import parser as dateparser
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import sys
from io import StringIO
from pathlib import Path
import platform

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    st.warning("VADER not available. Install: pip install vaderSentiment")

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MAX_UPLOAD_SIZE = 50 * 1024 * 1024
LOCK_TIMEOUT = 30

if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = GOOGLE_API_KEY

if "uploaded_chat_text" not in st.session_state:
    st.session_state.uploaded_chat_text = None

if "vader_analyzer" not in st.session_state:
    st.session_state.vader_analyzer = None

if "emotion_model" not in st.session_state:
    st.session_state.emotion_model = None

def cleanup_old_sessions(max_age_hours=6):
    try:
        now = time.time()
        for item in Path("analysis_data").glob("*"):
            if item.is_dir():
                if now - item.stat().st_mtime > max_age_hours * 3600:
                    shutil.rmtree(item)
        
        for item in Path(".").glob("vectorstore_*"):
            if item.is_dir():
                if now - item.stat().st_mtime > max_age_hours * 3600:
                    shutil.rmtree(item)
    except:
        pass

cleanup_old_sessions()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(int(time.time() * 1000000))

CHAT_FILE = "_chat.txt"
VECTOR_DIR = f"vectorstore_{st.session_state.session_id}"
ANALYSIS_DIR = f"analysis_data/{st.session_state.session_id}"
CHUNK_SIZE = 15

os.makedirs(ANALYSIS_DIR, exist_ok=True)


CHAT_PATTERN = re.compile(
    r"\[(\d{2}/\d{2}/\d{2}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)"
)


EMOTION_DIMENSIONS = {
    "valence": "How positive or negative the emotion is",
    "energized": "How energized or calm the emotion is",
    "dominance": "How in-control or submissive the emotion is",
    "intensity": "How strong the emotion is",
    "authenticity": "How genuine vs performative the expression seems"
}


def parse_whatsapp_chat(text):
    messages = []
    buffer = None

    for line in text.split("\n"):
        line = line.strip()
        match = CHAT_PATTERN.match(line)

        if match:
            if buffer:
                messages.append(buffer)

            date, time_, sender, message = match.groups()
            buffer = {
                "datetime": dateparser.parse(f"{date} {time_}", dayfirst=True),
                "sender": sender,
                "message": message
            }
        else:
            if buffer and line:
                buffer["message"] += " " + line

    if buffer:
        messages.append(buffer)

    return messages


def get_vader_analyzer():
    if st.session_state.vader_analyzer is not None:
        return st.session_state.vader_analyzer
    
    if not VADER_AVAILABLE:
        return None
    
    st.session_state.vader_analyzer = SentimentIntensityAnalyzer()
    return st.session_state.vader_analyzer


def get_emotion_classifier():
    if st.session_state.emotion_model is not None:
        return st.session_state.emotion_model
    
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        model = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            device=-1,
            top_k=1
        )
        st.session_state.emotion_model = model
        return model
    except Exception as e:
        st.warning(f"Could not load emotion model: {e}")
        return None


def calculate_message_importance(msg, idx, total_messages, day_boundaries, sender_stats):
    score = 0.0
    message = msg["message"]
    sender = msg["sender"]
    
    length = len(message)
    if length > 100:
        score += 0.3
    elif length > 50:
        score += 0.2
    elif length < 10:
        score -= 0.1
    
    if "?" in message:
        score += 0.25
    
    exclamation_count = message.count("!")
    if exclamation_count >= 2:
        score += 0.2
    elif exclamation_count == 1:
        score += 0.1
    
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]'
    emoji_count = len(re.findall(emoji_pattern, message))
    if emoji_count > 0:
        score += min(0.2, emoji_count * 0.05)
    
    is_day_boundary = any(abs(idx - b) <= 1 for b in day_boundaries)
    if is_day_boundary:
        score += 0.3
    
    sender_message_count = sender_stats.get(sender, {}).get("count", 1)
    if sender_message_count < total_messages * 0.2:
        score += 0.15
    
    if length > 10 and sum(1 for c in message if c.isupper()) / length > 0.5:
        score += 0.15
    
    if "http" in message.lower() or "www." in message.lower():
        score += 0.1
    
    if "@" in message:
        score += 0.1
    
    return min(1.0, max(0.0, score))


def smart_sample_messages(messages, target_count, base_sample_rate=None):
    total = len(messages)
    
    if target_count >= total:
        return messages, [1.0] * total
    
    sender_stats = {}
    for msg in messages:
        sender = msg["sender"]
        if sender not in sender_stats:
            sender_stats[sender] = {"count": 0, "total_length": 0}
        sender_stats[sender]["count"] += 1
        sender_stats[sender]["total_length"] += len(msg["message"])
    
    day_boundaries = []
    current_day = None
    for idx, msg in enumerate(messages):
        msg_day = msg["datetime"].date()
        if current_day is None or msg_day != current_day:
            if current_day is not None:
                day_boundaries.append(idx - 1)
            day_boundaries.append(idx)
            current_day = msg_day
    if messages:
        day_boundaries.append(len(messages) - 1)
    
    scored_messages = []
    for idx, msg in enumerate(messages):
        importance = calculate_message_importance(
            msg, idx, total, day_boundaries, sender_stats
        )
        
        if idx > 0:
            time_gap = (msg["datetime"] - messages[idx-1]["datetime"]).total_seconds() / 3600
            if time_gap > 4:
                importance += min(0.2, time_gap / 24)
        
        scored_messages.append({
            "index": idx,
            "message": msg,
            "importance": min(1.0, importance)
        })
    
    if len(scored_messages) <= target_count:
        selected = scored_messages
    else:
        threshold = 0.5
        high_importance = [m for m in scored_messages if m["importance"] >= threshold]
        
        remaining_slots = target_count - len(high_importance)
        
        if remaining_slots > 0:
            others = [m for m in scored_messages if m["importance"] < threshold]
            others_sorted = sorted(others, key=lambda x: x["importance"], reverse=True)
            selected = high_importance + others_sorted[:remaining_slots]
        else:
            selected = sorted(high_importance, key=lambda x: x["importance"], reverse=True)[:target_count]
        
        selected = sorted(selected, key=lambda x: x["index"])
    
    return [m["message"] for m in selected], [m["importance"] for m in selected]


def analyze_message_fast(message_text, vader_analyzer, emotion_model):
    result = {
        "message": message_text,
        "length": len(message_text),
    }
    
    if vader_analyzer:
        vader_scores = vader_analyzer.polarity_scores(message_text)
        result["vader_sentiment"] = {
            "compound": vader_scores["compound"],
            "positive": vader_scores["pos"],
            "negative": vader_scores["neg"],
            "neutral": vader_scores["neu"]
        }
        
        result["dimensions"] = {
            "valence": (vader_scores["compound"] + 1) / 2,
            "energized": 0.5,
            "dominance": 0.5,
            "intensity": abs(vader_scores["compound"]),
            "authenticity": 0.7
        }
    else:
        result["vader_sentiment"] = {"compound": 0, "positive": 0, "negative": 0, "neutral": 1}
        result["dimensions"] = {
            "valence": 0.5,
            "energized": 0.5,
            "dominance": 0.5,
            "intensity": 0.5,
            "authenticity": 0.5
        }
    
    if emotion_model and len(message_text.strip()) > 3:
        try:
            text_input = message_text[:512]
            emotion_result = emotion_model(text_input)
            if emotion_result and len(emotion_result) > 0:
                result["emotion_classifier"] = {
                    "label": emotion_result[0][0]["label"],
                    "score": emotion_result[0][0]["score"]
                }
                result["emotional_state"] = emotion_result[0][0]["label"].lower()
        except:
            result["emotional_state"] = "neutral"
    else:
        result["emotional_state"] = "neutral"
    
    result["linguistic_features"] = analyze_linguistic_patterns(message_text)
    
    features = result["linguistic_features"]
    energized_score = 0.5
    if features["exclamation_count"] > 0:
        energized_score += min(0.3, features["exclamation_count"] * 0.1)
    if features["caps_ratio"] > 0.3:
        energized_score += 0.2
    if features["emoji_count"] > 0:
        energized_score += 0.1
    result["dimensions"]["energized"] = min(1.0, energized_score)
    
    dominance_score = 0.5
    if features["question_marks"] > 0:
        dominance_score -= 0.15
    if features["statement_strength"] > 0.5:
        dominance_score += 0.2
    if features["caps_ratio"] > 0.4:
        dominance_score += 0.15
    result["dimensions"]["dominance"] = max(0.0, min(1.0, dominance_score))
    
    return result


def analyze_linguistic_patterns(text):
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]'
    
    features = {
        "length": len(text),
        "word_count": len(text.split()),
        "exclamation_count": text.count("!"),
        "question_marks": text.count("?"),
        "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "emoji_count": len(re.findall(emoji_pattern, text)),
        "url_count": len(re.findall(r'http[s]?://|www\.', text.lower())),
        "mention_count": text.count("@"),
        "ellipsis": text.count("...") + text.count("…"),
    }
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    features["sentence_count"] = len(sentences)
    features["avg_sentence_length"] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    
    uncertain_words = ["maybe", "perhaps", "might", "could", "possibly", "probably"]
    certain_words = ["definitely", "absolutely", "certainly", "must", "will", "always"]
    
    text_lower = text.lower()
    uncertain_count = sum(1 for w in uncertain_words if w in text_lower)
    certain_count = sum(1 for w in certain_words if w in text_lower)
    
    features["statement_strength"] = (certain_count - uncertain_count + 1) / 2
    
    informal_markers = ["lol", "lmao", "omg", "btw", "idk", "tbh", "nvm"]
    formal_markers = ["however", "therefore", "furthermore", "nevertheless", "regarding"]
    
    informal_count = sum(1 for m in informal_markers if m in text_lower)
    formal_count = sum(1 for m in formal_markers if m in text_lower)
    
    features["formality"] = "informal" if informal_count > formal_count else \
                           "formal" if formal_count > 0 else "neutral"
    
    return features


def fast_analyze_all_messages(messages, progress_callback=None):
    vader = get_vader_analyzer()
    emotion_model = get_emotion_classifier()
    
    if not vader:
        st.warning("VADER not available - analysis will be basic")
    if not emotion_model:
        st.warning("DistilBert not available")
    
    analyzed = []
    total = len(messages)
    
    for i, msg in enumerate(messages):
        if progress_callback and i % 100 == 0:
            progress_callback(i, total)
        
        analysis = analyze_message_fast(msg["message"], vader, emotion_model)
        
        analyzed.append({
            **msg,
            "fast_analysis": analysis
        })
    
    return analyzed


def identify_interesting_messages(analyzed_messages, top_n=100):
    scored = []
    
    for i, msg in enumerate(analyzed_messages):
        analysis = msg.get("fast_analysis", {})
        dims = analysis.get("dimensions", {})
        features = analysis.get("linguistic_features", {})
        
        interest_score = 0.0
        
        intensity = dims.get("intensity", 0)
        interest_score += intensity * 0.3
        
        valence = dims.get("valence", 0.5)
        valence_extremity = abs(valence - 0.5) * 2
        interest_score += valence_extremity * 0.2
        
        energized = dims.get("energized", 0.5)
        if energized > 0.7:
            interest_score += 0.15
        
        if features.get("length", 0) > 100:
            interest_score += 0.15
        
        if features.get("exclamation_count", 0) >= 2:
            interest_score += 0.1
        if features.get("question_marks", 0) >= 2:
            interest_score += 0.1
        
        if i > 0:
            prev_dims = analyzed_messages[i-1].get("fast_analysis", {}).get("dimensions", {})
            valence_shift = abs(dims.get("valence", 0.5) - prev_dims.get("valence", 0.5))
            if valence_shift > 0.4:
                interest_score += 0.2
        
        scored.append({
            "index": i,
            "message": msg,
            "interest_score": min(1.0, interest_score)
        })
    
    scored.sort(key=lambda x: x["interest_score"], reverse=True)
    top_interesting = scored[:top_n]
    
    top_interesting.sort(key=lambda x: x["index"])
    
    return [item["message"] for item in top_interesting]


def create_baseline_profile_llm(messages_sample, sender, llm):
    sample_size = min(50, len(messages_sample))
    step = max(1, len(messages_sample) // sample_size)
    sampled = [messages_sample[i] for i in range(0, len(messages_sample), step)][:sample_size]
    
    messages_text = "\n".join([
        f"{i+1}. {m['message'][:200]}"
        for i, m in enumerate(sampled)
    ])
    
    prompt = ChatPromptTemplate.from_template(
        """Analyze these {count} messages from "{sender}" to create a baseline profile.

MESSAGES:
{messages}

Return ONLY valid JSON:
{{
    "communication_style": {{
        "formality": "formal|casual|mixed",
        "verbosity": "concise|moderate|verbose",
        "expressiveness": "reserved|moderate|expressive",
        "humor_style": "none|dry|sarcastic|playful"
    }},
    "emotional_baseline": {{
        "default_tone": "their typical emotional tone",
        "common_emotions": ["list of common emotions"],
        "stress_indicators": ["what signals stress"],
        "excitement_indicators": ["what signals excitement"]
    }},
    "linguistic_patterns": {{
        "common_phrases": ["top phrases they use"],
        "languages": ["languages they use"]
    }}
}}"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "sender": sender,
            "count": len(sampled),
            "messages": messages_text
        })
        
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r'^```(?:json)?\s*', '', response)
            response = re.sub(r'\s*```$', '', response)
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        st.warning(f"Error creating baseline for {sender}: {e}")
    
    return {}


def deep_analyze_interesting_messages(interesting_messages, sender_profiles, llm):
    if not interesting_messages:
        return []
    
    batch_size = 20
    all_analyses = []
    
    for i in range(0, len(interesting_messages), batch_size):
        batch = interesting_messages[i:i+batch_size]
        
        batch_text = "\n".join([
            f"{j+1}. [{msg['sender']}] {msg['message'][:300]}"
            for j, msg in enumerate(batch)
        ])
        
        prompt = ChatPromptTemplate.from_template(
            """Analyze these {count} emotionally significant messages from a WhatsApp chat.

SENDER PROFILES:
{profiles}

MESSAGES:
{messages}

For each message, provide: emotional state, key insight (1 sentence).

Return JSON array with {count} objects:
[
  {{"emotional_state": "...", "insight": "..."}},
  ...
]"""
        )
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            profiles_text = "\n".join([
                f"- {s}: {p.get('emotional_baseline', {}).get('default_tone', 'N/A')}"
                for s, p in sender_profiles.items()
            ])
            
            response = chain.invoke({
                "count": len(batch),
                "profiles": profiles_text,
                "messages": batch_text
            })
            
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*', '', response)
                response = re.sub(r'\s*```$', '', response)
            
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                analyses = json.loads(json_match.group())
                all_analyses.extend(analyses[:len(batch)])
        except:
            all_analyses.extend([{"emotional_state": "unknown", "insight": ""} for _ in batch])
    
    return all_analyses


def aggregate_hybrid_stats(analyzed_messages, llm_insights):
    stats_per_person = defaultdict(lambda: {
        "total_messages": 0,
        "emotional_states": Counter(),
        "dimension_values": {dim: [] for dim in EMOTION_DIMENSIONS.keys()},
        "vader_sentiments": [],
        "linguistic_features": defaultdict(list),
        "hourly_distribution": Counter(),
        "daily_distribution": Counter(),
        "response_times": [],
    })
    
    sorted_messages = sorted(analyzed_messages, key=lambda x: x["datetime"])
    last_message_per_person = {}
    
    for msg in sorted_messages:
        sender = msg["sender"]
        stats = stats_per_person[sender]
        analysis = msg.get("fast_analysis", {})
        
        stats["total_messages"] += 1
        
        if "emotional_state" in analysis:
            stats["emotional_states"][analysis["emotional_state"]] += 1
        
        dimensions = analysis.get("dimensions", {})
        for dim, value in dimensions.items():
            if dim in stats["dimension_values"]:
                stats["dimension_values"][dim].append(value)
        
        if "vader_sentiment" in analysis:
            stats["vader_sentiments"].append(analysis["vader_sentiment"]["compound"])
        
        features = analysis.get("linguistic_features", {})
        for key, value in features.items():
            if isinstance(value, (int, float)):
                stats["linguistic_features"][key].append(value)
        
        stats["hourly_distribution"][msg["datetime"].hour] += 1
        stats["daily_distribution"][msg["datetime"].strftime("%A")] += 1
        
        if last_message_per_person:
            other_messages = [(p, dt) for p, dt in last_message_per_person.items() if p != sender]
            if other_messages:
                _, last_dt = max(other_messages, key=lambda x: x[1])
                response_time = (msg["datetime"] - last_dt).total_seconds() / 60
                if response_time < 1440:
                    stats["response_times"].append(response_time)
        
        last_message_per_person[sender] = msg["datetime"]
    
    profiles = {}
    for person, stats in stats_per_person.items():
        dimension_summary = {}
        for dim, values in stats["dimension_values"].items():
            if values:
                dimension_summary[dim] = {
                    "mean": round(np.mean(values), 3),
                    "std": round(np.std(values), 3)
                }
        
        profiles[person] = {
            "total_messages": stats["total_messages"],
            "dominant_emotional_state": stats["emotional_states"].most_common(1)[0][0] if stats["emotional_states"] else "neutral",
            "emotional_state_distribution": dict(stats["emotional_states"].most_common(10)),
            "emotional_dimension_profile": dimension_summary,
            "avg_sentiment": round(np.mean(stats["vader_sentiments"]), 3) if stats["vader_sentiments"] else 0.5,
            "sentiment_volatility": round(np.std(stats["vader_sentiments"]), 3) if stats["vader_sentiments"] else 0,
            "avg_response_time_min": round(np.mean(stats["response_times"]), 1) if stats["response_times"] else None,
            "most_active_hour": stats["hourly_distribution"].most_common(1)[0][0] if stats["hourly_distribution"] else None,
            "most_active_day": stats["daily_distribution"].most_common(1)[0][0] if stats["daily_distribution"] else None,
        }
    
    return profiles


def analyze_dynamics(analyzed_messages):
    interactions = defaultdict(list)
    sorted_messages = sorted(analyzed_messages, key=lambda x: x["datetime"])
    
    per_person_interactions = defaultdict(lambda: defaultdict(list))
    
    for i in range(1, len(sorted_messages)):
        prev_msg = sorted_messages[i-1]
        curr_msg = sorted_messages[i]
        
        if prev_msg["sender"] != curr_msg["sender"]:
            pair = tuple(sorted([prev_msg["sender"], curr_msg["sender"]]))
            
            prev_dims = prev_msg.get("fast_analysis", {}).get("dimensions", {})
            curr_dims = curr_msg.get("fast_analysis", {}).get("dimensions", {})
            
            alignment = {}
            for dim in EMOTION_DIMENSIONS.keys():
                if dim in prev_dims and dim in curr_dims:
                    alignment[dim] = 1 - abs(prev_dims[dim] - curr_dims[dim])
            
            interaction_data = {
                "timestamp": curr_msg["datetime"],
                "emotional_alignment": alignment,
                "response_time_sec": (curr_msg["datetime"] - prev_msg["datetime"]).total_seconds(),
                "responder": curr_msg["sender"],
                "initiator": prev_msg["sender"]
            }
            
            interactions[pair].append(interaction_data)
            per_person_interactions[curr_msg["sender"]][prev_msg["sender"]].append(interaction_data)
    
    relations = {}
    for pair, exchanges in interactions.items():
        avg_alignment = {}
        for dim in EMOTION_DIMENSIONS.keys():
            values = [e["emotional_alignment"].get(dim, 0.5) for e in exchanges if dim in e["emotional_alignment"]]
            if values:
                avg_alignment[dim] = round(np.mean(values), 3)
        
        response_times = [e["response_time_sec"] for e in exchanges if e["response_time_sec"] < 3600]
        
        person1_as_responder = [e for e in exchanges if e["responder"] == pair[0]]
        person2_as_responder = [e for e in exchanges if e["responder"] == pair[1]]
        
        person1_response_times = [e["response_time_sec"] for e in person1_as_responder if e["response_time_sec"] < 3600]
        person2_response_times = [e["response_time_sec"] for e in person2_as_responder if e["response_time_sec"] < 3600]
        
        relations[f"{pair[0]} ↔ {pair[1]}"] = {
            "total_exchanges": len(exchanges),
            "emotional_alignment": avg_alignment,
            "avg_response_time_sec": round(np.mean(response_times), 1) if response_times else None,
            "per_person_stats": {
                pair[0]: {
                    "times_responded": len(person1_as_responder),
                    "times_initiated": len(person2_as_responder),
                    "avg_response_time_sec": round(np.mean(person1_response_times), 1) if person1_response_times else None,
                    "response_rate": round(len(person1_as_responder) / len(exchanges), 3) if exchanges else 0
                },
                pair[1]: {
                    "times_responded": len(person2_as_responder),
                    "times_initiated": len(person1_as_responder),
                    "avg_response_time_sec": round(np.mean(person2_response_times), 1) if person2_response_times else None,
                    "response_rate": round(len(person2_as_responder) / len(exchanges), 3) if exchanges else 0
                }
            }
        }
    
    return relations


def save_analysis(data, filename):
    filepath = os.path.join(ANALYSIS_DIR, filename)
    lock_file = filepath + ".lock"
    
    try:
        if HAS_FCNTL and platform.system() != "Windows":
            with open(lock_file, 'w') as lf:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                    return True
                except IOError:
                    st.warning(f"File locked: {filename}. Try again in a moment.")
                    return False
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            return True
    except Exception as e:
        st.error(f"Error saving {filename}: {e}")
        return False


def load_analysis(filename):
    filepath = os.path.join(ANALYSIS_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    return None
                return json.loads(content)
        except:
            return None
    return None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


st.set_page_config(page_title="WhatsApp Chat RAG and Analyser", layout="wide")
st.title("WhatsApp Chat RAG and Analyser")
st.markdown("#### Clone and Use Offline Version for best performance and to sustain database between sessions.")
st.markdown("##### For security reasons, all data is expunged when session ends (that means if you hit refresh, you'd need to start over).")

st.sidebar.markdown("### Configuration")
api_key_input = st.sidebar.text_input(
    "Google API Key",
    value=st.session_state.google_api_key or "",
    type="password",
    help="Enter your Google API key or it will use the one from .env file"
)

if api_key_input:
    st.session_state.google_api_key = api_key_input
    GOOGLE_API_KEY = api_key_input

if not st.session_state.google_api_key:
    st.error("GOOGLE_API_KEY not found. Please enter your Google API key in the sidebar.")
    st.stop()

st.sidebar.markdown("### Chat File")
uploaded_file = st.sidebar.file_uploader(
    "Upload _chat.txt file",
    type=["txt"],
    help="Upload your WhatsApp chat export file (max 50MB)"
)

if uploaded_file:
    if uploaded_file.size > MAX_UPLOAD_SIZE:
        st.sidebar.error(f"File too large. Maximum size: {MAX_UPLOAD_SIZE / 1024 / 1024:.0f}MB")
        st.session_state.uploaded_chat_text = None
    else:
        try:
            st.session_state.uploaded_chat_text = uploaded_file.read().decode('utf-8')
            st.sidebar.success("File uploaded successfully")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)[:100]}")
            st.session_state.uploaded_chat_text = None

deps_ok = True
if not VADER_AVAILABLE:
    st.error("VADER not installed. Run: pip install vaderSentiment")
    deps_ok = False

if not TRANSFORMERS_AVAILABLE:
    st.warning("Transformers not installed. Install for classification: pip install transformers torch")

tabs = st.tabs([
    "Analysis Pipeline",
    "Profiles",
    "Insights",
    "Statistics",
    "RAG Query"
])


with tabs[0]:
    st.header("Analysis Pipeline")
    chat_text = None
    chat_source = "Unknown"
    
    if st.session_state.uploaded_chat_text:
        chat_text = st.session_state.uploaded_chat_text
        chat_source = "Uploaded File"
    elif os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            chat_text = f.read()
        chat_source = f"Local File ({CHAT_FILE})"
    
    if not chat_text:
        st.error(f"No chat file found. Please upload a _chat.txt file or place it in the working directory.")
        st.info("Use the file uploader in the sidebar to upload your WhatsApp chat export.")
    elif not deps_ok:
        st.error("Please install required dependencies first")
    else:
        st.info(f"Using chat from: {chat_source}")
        
        messages = parse_whatsapp_chat(chat_text)
        st.success(f"Loaded {len(messages):,} messages")
        
        senders = list(set(m["sender"] for m in messages))
        st.info(f"Participants: {', '.join(senders)}")
        
        st.subheader("Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_sampling = st.checkbox(
                "Enable smart sampling",
                value=len(messages) > 10000,
                help="For very large chats, sample most important messages"
            )
            
            if use_sampling:
                sample_target = st.slider(
                    "Target messages for analysis",
                    1000, min(20000, len(messages)),
                    min(5000, len(messages)),
                    500
                )
            else:
                sample_target = len(messages)
            
            interesting_count = st.slider(
                "Important messages for deep LLM analysis",
                10, min(500, len(messages)), min(100, len(messages)), 10,
                help="Top N most important messages get detailed LLM analysis"
            )
        
        with col2:
            st.metric("Messages to analyze (fast NLP)", f"{sample_target:,}")
            st.metric("Messages for LLM analysis", interesting_count)
            
            fast_time = sample_target / 100
            llm_time = (len(senders) + interesting_count / 20) * 3
            total_time = (fast_time + llm_time) / 60
            
            st.metric("Estimated time", f"~{total_time:.1f} minutes")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run Analysis", type="primary", use_container_width=True):
                progress_container = st.container()
                console_container = st.container()
                
                with progress_container:
                    overall_progress = st.progress(0, text="Starting...")
                    stage_status = st.empty()
                    detail_status = st.empty()
                    time_status = st.empty()

                console_placeholder = console_container.empty()
                console_logs = []
                last_update = [0]
                
                class ConsoleLogger:
                    def __init__(self, original_stream):
                        self.original_stream = original_stream
                        self.buffer = []
                    
                    def write(self, message):
                        self.original_stream.write(message)
                        if message.strip():
                            self.buffer.append(message.strip())
                            console_logs.append(message.strip())
                            current_time = time.time()
                            if current_time - last_update[0] > 0.5 and console_logs:
                                last_update[0] = current_time
                                console_placeholder.empty()
                                with console_placeholder.container():
                                    st.markdown("### Console Output")
                                    console_text = "\n".join(console_logs[-50:])
                                    st.markdown(
                                        f"""
                                        <div style="max-height: 200px; overflow-y: auto; border: 1px solid var(--border-color, #e0e0e0); border-radius: 4px; padding: 8px; background-color: var(--background-color); color: var(--text-color); font-family: monospace; font-size: 12px;">
                                        <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;">{console_text}</pre>
                                        </div>
                                        <style>
                                        @media (prefers-color-scheme: dark) {{
                                            :root {{
                                                --border-color: #444;
                                                --background-color: #1e1e1e;
                                                --text-color: #e0e0e0;
                                            }}
                                        }}
                                        @media (prefers-color-scheme: light) {{
                                            :root {{
                                                --border-color: #e0e0e0;
                                                --background-color: #f5f5f5;
                                                --text-color: #333;
                                            }}
                                        }}
                                        </style>
                                        """,
                                        unsafe_allow_html=True
                                    )
                    
                    def flush(self):
                        self.original_stream.flush()
                    
                    def isatty(self):
                        return self.original_stream.isatty()

                old_stdout = sys.stdout
                old_stderr = sys.stderr
                logger = ConsoleLogger(old_stdout)
                sys.stdout = logger
                sys.stderr = logger
                
                try:
                    import time as time_module
                    start_time = time_module.time()
                    
                    if use_sampling:
                        stage_status.markdown("### Stage 1/4: Smart Sampling")
                        overall_progress.progress(0.1, text="Sampling...")
                        
                        sampled_messages, importance_scores = smart_sample_messages(
                            messages, sample_target
                        )
                        detail_status.success(f"Selected {len(sampled_messages):,}/{len(messages):,} messages")
                    else:
                        sampled_messages = messages
                    
                    stage_status.markdown("### Stage 2/4: NLP Analysis (All Messages)")
                    overall_progress.progress(0.2, text="NLP analysis...")
                    detail_status.text("Running VADER + classifier + pattern matching...")
                    
                    def fast_progress(current, total):
                        pct = current / total
                        overall_progress.progress(0.2 + pct * 0.4, text=f"Analyzing: {current:,}/{total:,}")
                        elapsed = time_module.time() - start_time
                        rate = current / elapsed if elapsed > 0 else 0
                        time_status.info(f"Speed: {rate:.0f} msg/sec | Elapsed: {elapsed:.1f}s")
                    
                    analyzed = fast_analyze_all_messages(sampled_messages, fast_progress)
                    
                    save_analysis(analyzed, "analyzed_messages.json")
                    
                    fast_time = time_module.time() - start_time
                    detail_status.success(f"Analysis complete: {len(analyzed):,} messages in {fast_time:.1f}s ({len(analyzed)/fast_time:.0f} msg/sec)")
                    
                    stage_status.markdown("### Stage 3/4: LLM Baseline Profiles")
                    overall_progress.progress(0.6, text="Creating profiles with LLM...")
                    
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-flash-latest",
                        temperature=0.3,
                        google_api_key=st.session_state.google_api_key
                    )
                    
                    messages_by_sender = defaultdict(list)
                    for msg in analyzed:
                        messages_by_sender[msg["sender"]].append(msg)
                    
                    baseline_profiles = {}
                    for idx, sender in enumerate(senders):
                        detail_status.text(f"LLM profiling: {sender} ({idx+1}/{len(senders)})")
                        profile = create_baseline_profile_llm(
                            messages_by_sender[sender],
                            sender,
                            llm
                        )
                        baseline_profiles[sender] = profile
                    
                    save_analysis(baseline_profiles, "baseline_profiles.json")
                    detail_status.success(f"Created {len(baseline_profiles)} baseline profiles")
                    
                    stage_status.markdown("### Stage 4/4: LLM Analysis of Key Moments")
                    overall_progress.progress(0.8, text="Analysis of important messages...")
                    
                    detail_status.text(f"Identifying top {interesting_count} important messages...")
                    interesting_msgs = identify_interesting_messages(analyzed, interesting_count)
                    
                    detail_status.text(f"Running LLM analysis on {len(interesting_msgs)} messages...")
                    llm_insights = deep_analyze_interesting_messages(
                        interesting_msgs,
                        baseline_profiles,
                        llm
                    )
                    
                    save_analysis({
                        "interesting_messages": interesting_msgs,
                        "llm_insights": llm_insights
                    }, "llm_insights.json")
                    
                    detail_status.success(f"Analysis of {len(interesting_msgs)} key moments complete")
                    
                    overall_progress.progress(0.9, text="Aggregating insights...")
                    detail_status.text("Building final profiles...")
                    
                    profiles = aggregate_hybrid_stats(analyzed, llm_insights)
                    save_analysis(profiles, "advanced_profiles.json")
                    
                    detail_status.text("Analyzing relations...")
                    relations = analyze_dynamics(analyzed)
                    save_analysis(relations, "relations.json")
                    
                    total_time = time_module.time() - start_time
                    overall_progress.progress(1.0, text="Complete!")
                    
                    time_status.success(
                        f"Total time: {total_time/60:.2f} minutes | "
                        f"Analyzed {len(analyzed):,} messages ({len(analyzed)/total_time:.0f} msg/sec)"
                    )
                    
                    st.balloons()
                    st.success(f"""
                    **Analysis Complete**
                    - NLP: {len(analyzed):,} messages
                    - LLM profiles: {len(baseline_profiles)} people
                    - Analysis: {len(interesting_msgs)} key moments
                    - Total time: {total_time/60:.2f} minutes
                    - Speed: {len(analyzed)/total_time:.0f} msg/sec
                    """)
                    
                except Exception as e:
                    overall_progress.progress(0, text="Error")
                    detail_status.error(f"Error: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
                
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
        
        with col2:
            if st.button("Build RAG Index", use_container_width=True):
                analyzed = load_analysis("analyzed_messages.json")
                if not analyzed:
                    st.error("Run Analysis first")
                else:
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    try:
                        docs = []
                        for m in analyzed:
                            text = f"[{m['datetime']}] {m['sender']}: {m['message']}"
                            analysis = m.get('fast_analysis', {})
                            docs.append(Document(
                                page_content=text,
                                metadata={
                                    "sender": m["sender"],
                                    "emotion": analysis.get("emotional_state", "neutral"),
                                    "sentiment": analysis.get("vader_sentiment", {}).get("compound", 0),
                                    "intensity": analysis.get("dimensions", {}).get("intensity", 0)
                                }
                            ))
                        
                        chunks = []
                        for i in range(0, len(docs), CHUNK_SIZE):
                            text = "\n".join(d.page_content for d in docs[i:i+CHUNK_SIZE])
                            chunks.append(Document(page_content=text))
                        
                        status.text("Creating embeddings...")
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/text-embedding-004",
                            google_api_key=st.session_state.google_api_key
                        )
                        vs = FAISS.from_documents(chunks, embeddings)
                        vs.save_local(VECTOR_DIR)
                        
                        progress_bar.progress(1.0)
                        status.empty()
                        st.success("RAG index ready")
                        
                    except Exception as e:
                        status.empty()
                        st.error(f"Error building index: {str(e)}")


with tabs[1]:
    st.header("Personality Profiles")
    
    profiles = load_analysis("advanced_profiles.json")
    baseline_profiles = load_analysis("baseline_profiles.json")
    
    if not profiles:
        st.warning("Run analysis first")
    else:
        selected = st.selectbox("Select person", sorted(profiles.keys()))
        
        if selected:
            profile = profiles[selected]
            baseline = baseline_profiles.get(selected, {}) if baseline_profiles else {}
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Messages", f"{profile['total_messages']:,}")
            with col2:
                st.metric("Dominant Emotion", profile['dominant_emotional_state'].title())
            with col3:
                st.metric("Avg Sentiment", f"{profile['avg_sentiment']:.3f}")
            
            if baseline:
                st.subheader("Communication Style (LLM Profile)")
                st.json(baseline.get("communication_style", {}))
                
                st.subheader("Emotional Baseline (LLM Profile)")
                st.json(baseline.get("emotional_baseline", {}))
            
            st.subheader("Emotional Dimensions (Fast NLP)")
            dim_prof = profile.get('emotional_dimension_profile', {})
            if dim_prof:
                fig_data = {dim.title(): vals['mean'] for dim, vals in dim_prof.items()}
                st.bar_chart(fig_data)


with tabs[2]:
    st.header("Insights")
    
    analyzed = load_analysis("analyzed_messages.json")
    insights_data = load_analysis("llm_insights.json")
    
    if not analyzed:
        st.warning("Run analysis first")
    else:
        all_emotions = [m["fast_analysis"]["emotional_state"] 
                       for m in analyzed 
                       if "fast_analysis" in m]
        emotion_counts = Counter(all_emotions)
        
        st.subheader("Emotional State Distribution (Fast NLP)")
        st.bar_chart(dict(emotion_counts.most_common(15)))
        
        if insights_data and "llm_insights" in insights_data:
            st.subheader("Key Moments")
            st.info(f"Analyzed {len(insights_data['llm_insights'])} most interesting messages with LLM")
            
            interesting_msgs = insights_data.get("interesting_messages", [])
            llm_insights = insights_data.get("llm_insights", [])
            
            from datetime import datetime

            for i, (msg, insight) in enumerate(list(zip(interesting_msgs, llm_insights))[:10]):
                try:
                    if not isinstance(msg['datetime'], datetime):
                        msg['datetime'] = datetime.strptime(msg['datetime'], '%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    st.error(f"Error parsing datetime: {e}")
                    continue

                with st.expander(f"{msg['sender']} - {insight.get('emotional_state', 'N/A').title()}"):
                    st.write(f"**Time:** {msg['datetime'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Message:** {msg['message'][:200]}")
                    st.write(f"**Insight:** {insight.get('insight', 'N/A')}")


with tabs[3]:
    st.header("Statistics")
    
    relations = load_analysis("relations.json")
    
    if not relations:
        st.warning("Run analysis first")
    else:
        for pair, metrics in relations.items():
            with st.expander(f"{pair}"):
                st.metric("Total Exchanges", metrics['total_exchanges'])
                
                st.subheader("Per-Person Metrics")
                
                per_person = metrics.get('per_person_stats', {})
                if per_person:
                    cols = st.columns(2)
                    
                    for idx, (person, stats) in enumerate(per_person.items()):
                        with cols[idx]:
                            st.markdown(f"**{person}**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Times Responded", stats['times_responded'])
                                st.metric("Response Rate", f"{stats['response_rate']:.1%}")
                            with col2:
                                st.metric("Times Initiated", stats['times_initiated'])
                                if stats['avg_response_time_sec']:
                                    st.metric("Avg Response Time", f"{stats['avg_response_time_sec']/60:.1f}m")
                
                st.subheader("Alignment")
                alignment = metrics.get('emotional_alignment', {})
                if alignment:
                    st.bar_chart(alignment)


with tabs[4]:
    st.header("RAG-Powered Query")
    
    st.markdown("""
    Use RAG to find illustrative examples of specific moments.
    """)
    
    if not os.path.exists(VECTOR_DIR):
        st.warning("Build RAG index first in the Analysis Pipeline tab")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=st.session_state.google_api_key
        )
        vectorstore = FAISS.load_local(
            VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0,
            google_api_key=st.session_state.google_api_key
        )
        
        prompt = ChatPromptTemplate.from_template(
            """You are analyzing a WhatsApp chat.

Context (with timestamps and senders):
{context}

Question: {question}

Provide specific examples from the context with timestamps and explain the emotional patterns."""
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        st.subheader("Suggested Queries")
        suggestions = [
            "Show examples where someone sounded anxious or stressed",
            "Find moments of high excitement or joy",
            "When did emotional tone shift negatively?",
            "Find examples of conflict or frustration",
            "Show supportive or empathetic messages"
        ]
        
        selected_suggestion = st.selectbox("Or pick a suggestion:", ["Custom query"] + suggestions)
        
        if selected_suggestion != "Custom query":
            query = st.text_input("Query:", value=selected_suggestion)
        else:
            query = st.text_input("Ask about patterns or specific moments:")
        
        if query:
            with st.spinner("Searching..."):
                response = rag_chain.invoke(query)
            
            st.markdown("### Analysis")
            st.write(response)


def cleanup_session_resources():
    try:
        if os.path.exists(VECTOR_DIR):
            shutil.rmtree(VECTOR_DIR)
    except:
        pass
    
    try:
        lock_files = Path(ANALYSIS_DIR).glob("*.lock")
        for lock_file in lock_files:
            lock_file.unlink()
    except:
        pass


atexit.register(cleanup_session_resources)