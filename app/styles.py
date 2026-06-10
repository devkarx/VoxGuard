"""
CSS styles for the Streamlit Deepfake Audio Detector app.

All styling is centralized here for maintainability.
Uses Inter font, dark glassmorphism theme, and gradient accents.
"""

MAIN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* --- Global --- */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* --- Hero Header --- */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero .tagline {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.4rem;
    }

    /* --- Glass Card --- */
    .glass-card {
        background: rgba(30, 32, 44, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.6rem;
        margin: 0.8rem 0;
    }

    /* --- Result Cards --- */
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.2rem 0;
        backdrop-filter: blur(16px);
        transition: transform 0.2s ease;
    }
    .result-genuine {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.10) 0%, rgba(52, 211, 153, 0.06) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .result-deepfake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.10) 0%, rgba(248, 113, 113, 0.06) 100%);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .result-card h2 {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.01em;
    }
    .result-card .conf {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.85;
    }

    /* --- Model Badge --- */
    .model-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.3rem 0.9rem;
        border-radius: 100px;
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: rgba(129, 140, 248, 0.15);
        border: 1px solid rgba(129, 140, 248, 0.3);
        color: #a5b4fc;
    }

    /* --- Stat Box --- */
    .stat-box {
        text-align: center;
        padding: 1.2rem 1rem;
        border-radius: 14px;
        background: rgba(30, 32, 44, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stat-box .label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    .stat-box .value {
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    /* --- Colors --- */
    .text-green  { color: #34d399; }
    .text-red    { color: #f87171; }
    .text-purple { color: #a78bfa; }
    .text-slate  { color: #94a3b8; }

    /* --- Section Labels --- */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.6rem 0;
    }

    /* --- Sidebar --- */
    .sidebar-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 1.2rem 0 0.5rem 0;
    }
    .sidebar-text {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.6;
    }
</style>
"""
