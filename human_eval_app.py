import streamlit as st
import pandas as pd
import hashlib
import os
import base64
import io
import requests

DATA_PATH = os.path.join(os.path.dirname(__file__), "human_eval_data.csv")

COLS = ["pair_index", "post_1", "post_2", "post_key", "label", "start_ind"]
ANNOTATIONS_DIR = "annotation_app/annotations"


def _headers():
    return {
        "Authorization": f"Bearer {st.secrets['github_token']}",
        "Accept": "application/vnd.github+json",
    }


def _api_url(path_in_repo):
    return f"https://api.github.com/repos/{st.secrets['github_repo']}/contents/{path_in_repo}"


def _annotation_path(name):
    return f"{ANNOTATIONS_DIR}/annotations_{name}.csv"


def _fetch_file(path_in_repo):
    r = requests.get(_api_url(path_in_repo), headers=_headers())
    if r.status_code == 404:
        return pd.DataFrame(columns=COLS), None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return pd.read_csv(io.StringIO(content)), data["sha"]


def load_existing(name):
    df, _ = _fetch_file(_annotation_path(name))
    return df


def get_num_annotators():
    r = requests.get(_api_url(ANNOTATIONS_DIR), headers=_headers())
    if r.status_code == 404:
        return 0
    r.raise_for_status()
    return len(r.json())


def load_data(annotator_name):
    df_all = pd.read_csv(DATA_PATH, index_col=0).reset_index(drop=True)
    existing = load_existing(annotator_name)
    if not existing.empty:
        start_ind = int(existing["start_ind"].iloc[0])
    else:
        start_ind = 28 * (get_num_annotators() % 4)
    df = df_all[start_ind: start_ind + 28].reset_index(drop=True)
    seed = int(hashlib.md5(annotator_name.encode()).hexdigest(), 16) % (2**32)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True), start_ind, existing


def save_response(name, row, label, start_ind):
    path = _annotation_path(name)
    existing, sha = _fetch_file(path)
    new_row = pd.DataFrame([{
        "pair_index": int(row.name),
        "post_1": row["post_1"],
        "post_2": row["post_2"],
        "post_key": row["post_key"],
        "label": label,
        "start_ind": start_ind,
    }])
    updated = pd.concat([existing, new_row], ignore_index=True)
    payload = {
        "message": f"annotation: {name}",
        "content": base64.b64encode(updated.to_csv(index=False).encode()).decode(),
    }
    if sha:
        payload["sha"] = sha
    requests.put(_api_url(path), headers=_headers(), json=payload).raise_for_status()


# ── Login screen ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Human Eval", layout="wide")

if "annotator" not in st.session_state:
    st.title("Naturalism Evaluation")
    name = st.text_input("Enter your Prolific ID to begin:")
    if st.button("Start") and name.strip():
        st.session_state.annotator = name.strip()
        st.rerun()
    st.stop()

annotator = st.session_state.annotator
df, start_ind, done = load_data(annotator)
done_indices = set(done["pair_index"].tolist())

remaining = [i for i in range(len(df)) if i not in done_indices]
n_total = len(df)
n_done = len(done_indices)

# ── Sidebar progress ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"**Annotator:** {annotator}")
    st.progress(n_done / n_total)
    st.markdown(f"{n_done} / {n_total} done")
    if st.button("Switch annotator"):
        del st.session_state["annotator"]
        st.rerun()

# ── Done screen ───────────────────────────────────────────────────────────────

if not remaining:
    st.success(f"All {n_total} pairs complete! Thank you for your annotations.")
    st.link_button("Complete on Prolific", st.secrets["prolific_completion_url"])
    st.stop()

# ── Annotation screen ─────────────────────────────────────────────────────────

curr_idx = remaining[0]
row = df.iloc[curr_idx]

st.markdown(f"### Pair {n_done + 1} of {n_total}")
st.markdown("**Reflect upon your own experiences to evaluate the naturalness of these two posts: which one of these posts are you or your peers more likely to write? Judge the wording, structure, and writing style of the posts.**")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Post 1")
    st.markdown(row["post_1"])
with col2:
    st.markdown("#### Post 2")
    st.markdown(row["post_2"])

st.markdown("---")

bcol1, bcol2, bcol3 = st.columns(3)
with bcol1:
    if st.button("Post 1 is more natural", use_container_width=True):
        save_response(annotator, row, "post1", start_ind)
        st.rerun()
with bcol2:
    if st.button("Equally natural", use_container_width=True):
        save_response(annotator, row, "equal", start_ind)
        st.rerun()
with bcol3:
    if st.button("Post 2 is more natural", use_container_width=True):
        save_response(annotator, row, "post2", start_ind)
        st.rerun()
