"""Small, accessible style layer for the Streamlit chat workspace."""

APP_CSS = """
<style>
:root {
    --chat-accent: #7c9cff;
    --chat-border: rgba(148, 163, 184, 0.18);
}
.stApp {
    background:
      radial-gradient(circle at 82% 2%, rgba(91, 119, 255, .10), transparent 28rem),
      var(--background-color);
}
.block-container {
    max-width: 1120px;
    padding-top: 2.0rem;
    padding-bottom: 7rem;
}
.chat-hero {
    border: 1px solid var(--chat-border);
    border-radius: 22px;
    padding: 1.35rem 1.55rem;
    margin-bottom: 1.2rem;
    background: linear-gradient(115deg, rgba(79, 104, 201, .16), rgba(255,255,255,.025));
}
.chat-hero h1 {
    font-size: clamp(1.7rem, 3vw, 2.35rem);
    letter-spacing: -.045em;
    margin: 0 0 .3rem 0;
}
.chat-hero p {
    color: var(--secondary-text-color);
    margin: 0;
    font-size: .96rem;
}
.eyebrow {
    color: var(--chat-accent);
    font-size: .72rem;
    font-weight: 750;
    letter-spacing: .13em;
    text-transform: uppercase;
    margin-bottom: .5rem;
}
[data-testid="stSidebar"] {
    border-right: 1px solid var(--chat-border);
}
[data-testid="stChatMessage"] {
    border: 1px solid var(--chat-border);
    border-radius: 16px;
    padding: .95rem 1rem;
    margin-bottom: .75rem;
}
[data-testid="stChatInput"] {
    border-color: var(--chat-border);
}
div[data-testid="stStatusWidget"] {
    border: 1px solid var(--chat-border);
    border-radius: 12px;
}
.small-muted {
    color: var(--secondary-text-color);
    font-size: .82rem;
}
.source-card {
    border-left: 3px solid var(--chat-accent);
    padding: .25rem .75rem;
    margin: .55rem 0;
}
</style>
"""
