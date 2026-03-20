import { useState, useEffect, useRef } from "react";

const SECTIONS = ["About", "Tracks", "Prizes", "Timeline", "Partners"];
const CONTACT_EMAIL = "eugene.shcherbinin@bloomsburytech.com";

function Reveal({ children, delay = 0 }) {
  const ref = useRef(null);
  const [v, setV] = useState(false);
  useEffect(() => {
    const o = new IntersectionObserver(([e]) => { if (e.isIntersecting) setV(true); }, { threshold: 0.1 });
    if (ref.current) o.observe(ref.current);
    return () => o.disconnect();
  }, []);
  return <div ref={ref} style={{ opacity: v ? 1 : 0, transform: v ? "translateY(0)" : "translateY(20px)", transition: `opacity 0.5s ease ${delay}ms, transform 0.5s ease ${delay}ms` }}>{children}</div>;
}

export default function Site() {
  const [scrollY, setScrollY] = useState(0);
  const [dark, setDark] = useState(true);

  useEffect(() => {
    const fn = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", fn, { passive: true });
    return () => window.removeEventListener("scroll", fn);
  }, []);

  const go = (id) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });

  const t = dark ? "dark" : "light";

  return (
    <div data-theme={t} style={{ minHeight: "100vh" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400;1,500&family=EB+Garamond:ital,wght@0,400;0,500;1,400;1,500&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ═══ LIGHT — Bloomsbury White ═══ */
        [data-theme="light"] {
          --bg: #FAFAF8;
          --bg-card: #FFFFFF;
          --bg-card-hover: #F3F2EF;
          --text: #111111;
          --text-dim: #6B6560;
          --text-label: #8A8078;
          --border: rgba(0,0,0,0.06);
          --border-hover: rgba(0,0,0,0.13);
          --accent: #111111;
          --accent-dim: rgba(0,0,0,0.05);
          --btn-bg: #111111;
          --btn-text: #FAFAF8;
          --nav-bg: rgba(250,250,248,0.92);
          --serif: 'EB Garamond', Georgia, serif;
          --glass-border: rgba(0,0,0,0.06);
          background: #FAFAF8;
          color: #111111;
        }

        /* ═══ DARK — Theos Dashboard ═══ */
        [data-theme="dark"] {
          --bg: #0A0A0F;
          --bg-card: rgba(15, 15, 25, 0.85);
          --bg-card-hover: rgba(20, 20, 35, 0.9);
          --text: #E8E6E3;
          --text-dim: rgba(232, 230, 227, 0.5);
          --text-label: rgba(232, 230, 227, 0.35);
          --border: rgba(212, 168, 67, 0.1);
          --border-hover: rgba(212, 168, 67, 0.2);
          --accent: #D4A843;
          --accent-dim: rgba(212, 168, 67, 0.08);
          --btn-bg: #D4A843;
          --btn-text: #0A0A0F;
          --nav-bg: rgba(10, 10, 15, 0.92);
          --serif: 'Cormorant Garamond', Georgia, serif;
          --glass-border: rgba(212, 168, 67, 0.15);
          background: #0A0A0F;
          color: #E8E6E3;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        [data-theme] {
          font-family: 'Inter', system-ui, sans-serif;
          -webkit-font-smoothing: antialiased;
          overflow-x: hidden;
          transition: background 0.5s ease, color 0.5s ease;
        }
        ::selection { background: var(--accent); color: var(--bg); }
        a { color: inherit; text-decoration: none; }

        /* NAV */
        .nav {
          position: fixed; top: 0; left: 0; right: 0; z-index: 100;
          padding: 22px 48px;
          display: flex; align-items: center; justify-content: space-between;
          transition: all 0.35s ease;
        }
        .nav.scrolled {
          background: var(--nav-bg);
          backdrop-filter: blur(20px);
          border-bottom: 1px solid var(--glass-border);
          padding: 14px 48px;
        }
        .nav-logo {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px; letter-spacing: 3px;
          text-transform: uppercase; font-weight: 500;
        }
        [data-theme="dark"] .nav-logo { color: var(--accent); }
        .nav-right { display: flex; align-items: center; gap: 28px; }
        .nav-links { display: flex; gap: 28px; list-style: none; }
        .nav-links a {
          color: var(--text-dim); font-size: 12px; font-weight: 400;
          letter-spacing: 0.4px; cursor: pointer; transition: color 0.3s;
        }
        .nav-links a:hover { color: var(--text); }
        [data-theme="dark"] .nav-links a:hover { color: var(--accent); }
        .nav-btn {
          background: transparent; color: var(--text);
          border: 1px solid var(--border-hover);
          padding: 6px 14px; font-family: 'JetBrains Mono', monospace;
          font-size: 11px; cursor: pointer; transition: all 0.3s; letter-spacing: 0.5px;
        }
        .nav-btn:hover { background: var(--btn-bg); color: var(--btn-text); border-color: var(--btn-bg); }

        /* HERO */
        .hero {
          min-height: 100vh; display: flex; flex-direction: column;
          justify-content: center; padding: 140px 48px 100px; max-width: 860px;
        }
        .hero-tag {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px; letter-spacing: 3px;
          text-transform: uppercase; color: var(--text-label); margin-bottom: 36px;
        }
        [data-theme="dark"] .hero-tag { color: var(--accent); opacity: 0.7; }
        .hero h1 {
          font-family: var(--serif);
          font-size: clamp(42px, 6vw, 74px);
          line-height: 1.06; font-weight: 400; letter-spacing: -0.5px;
        }
        .hero h1 em { font-style: italic; }
        [data-theme="dark"] .hero h1 em { color: var(--accent); }
        .hero-sub {
          margin-top: 28px; font-size: 15px; line-height: 1.8;
          color: var(--text-dim); max-width: 480px; font-weight: 300;
        }
        .hero-meta { margin-top: 48px; display: flex; gap: 48px; flex-wrap: wrap; }
        .hero-meta-item { display: flex; flex-direction: column; gap: 5px; }
        .hero-meta-label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; letter-spacing: 2px;
          text-transform: uppercase; color: var(--text-label);
        }
        .hero-meta-value { font-family: var(--serif); font-size: 19px; }
        .hero-actions { margin-top: 48px; display: flex; gap: 12px; flex-wrap: wrap; }

        .btn-p {
          background: var(--btn-bg); color: var(--btn-text);
          border: none; padding: 12px 32px;
          font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500;
          cursor: pointer; transition: all 0.3s; letter-spacing: 0.2px;
        }
        .btn-p:hover { opacity: 0.85; transform: translateY(-1px); }
        .btn-o {
          background: transparent; color: var(--text);
          border: 1px solid var(--border-hover); padding: 12px 32px;
          font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 400;
          cursor: pointer; transition: all 0.3s;
        }
        .btn-o:hover { border-color: var(--accent); color: var(--accent); }
        .hero-email {
          font-family: 'JetBrains Mono', monospace;
          font-size: 14px; letter-spacing: 0.3px;
          color: var(--text); border-bottom: 1px solid var(--border-hover);
          padding-bottom: 2px; transition: all 0.3s;
        }
        .hero-email:hover { color: var(--accent); border-color: var(--accent); }

        /* SECTIONS */
        .sec { padding: 100px 48px; max-width: 1060px; margin: 0 auto; }
        .sec-num {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px; letter-spacing: 2px;
          color: var(--text-label); margin-bottom: 18px;
        }
        [data-theme="dark"] .sec-num { color: var(--accent); opacity: 0.6; }
        .sec-title {
          font-family: var(--serif);
          font-size: clamp(28px, 3.8vw, 44px);
          line-height: 1.12; font-weight: 400;
          margin-bottom: 12px; max-width: 560px;
        }
        .sec-desc {
          font-size: 15px; line-height: 1.8; color: var(--text-dim);
          max-width: 480px; font-weight: 300; margin-bottom: 48px;
        }

        .div { height: 1px; background: var(--border); max-width: 1060px; margin: 0 auto; }

        /* CARDS */
        .cg {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
          gap: 1px; background: var(--border); border: 1px solid var(--border);
        }
        .cd {
          background: var(--bg); padding: 36px 28px; transition: background 0.3s;
        }
        [data-theme="dark"] .cd { background: var(--bg-card); backdrop-filter: blur(12px); }
        .cd:hover { background: var(--bg-card-hover); }
        .cd-icon {
          font-size: 17px; margin-bottom: 16px;
          color: var(--text-label); font-family: 'JetBrains Mono', monospace;
        }
        [data-theme="dark"] .cd-icon { color: var(--accent); opacity: 0.7; }
        .cd h3 {
          font-family: var(--serif); font-size: 20px;
          font-weight: 400; margin-bottom: 10px;
        }
        .cd p { font-size: 13px; line-height: 1.75; color: var(--text-dim); font-weight: 300; }
        .cd-tag {
          display: inline-block; margin-top: 12px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; letter-spacing: 1.5px;
          text-transform: uppercase; color: var(--text-label);
        }

        /* ARCH */
        .arch {
          border: 1px solid var(--border); padding: 40px 24px;
          margin-bottom: 18px; text-align: center;
        }
        [data-theme="dark"] .arch { background: var(--bg-card); backdrop-filter: blur(12px); border-color: var(--glass-border); }
        .arch-label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; letter-spacing: 3px;
          color: var(--text-label); text-transform: uppercase; margin-bottom: 22px;
        }
        .arch-row { display: flex; justify-content: center; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 22px; }
        .arch-pill {
          padding: 6px 14px; border: 1px solid var(--border-hover);
          font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.3px;
        }
        [data-theme="dark"] .arch-pill { border-color: var(--glass-border); color: var(--accent); }
        .arch-plus { color: var(--text-label); font-size: 13px; }
        .arch-arr { color: var(--text-label); font-size: 16px; margin-bottom: 16px; }
        .arch-out {
          display: inline-block; padding: 10px 24px;
          border: 1px solid var(--text);
          font-family: var(--serif); font-size: 16px; font-style: italic;
        }
        [data-theme="dark"] .arch-out { border-color: var(--accent); color: var(--accent); }

        /* PRIZES */
        .pg {
          display: grid; grid-template-columns: repeat(3, 1fr);
          gap: 1px; background: var(--border); border: 1px solid var(--border);
        }
        .pr { background: var(--bg); padding: 40px 26px; }
        [data-theme="dark"] .pr { background: var(--bg-card); }
        .pr.ft { background: var(--bg-card-hover); }
        .pr-pl {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; letter-spacing: 3px;
          text-transform: uppercase; color: var(--text-label); margin-bottom: 12px;
        }
        .pr-am { font-family: var(--serif); font-size: 40px; margin-bottom: 22px; }
        .pr-am .c { font-size: 18px; color: var(--text-dim); }
        [data-theme="dark"] .pr-am { color: var(--accent); }
        .pr-pk { list-style: none; }
        .pr-pk li {
          font-size: 13px; color: var(--text-dim); padding: 6px 0;
          border-bottom: 1px solid var(--border); font-weight: 300;
        }

        /* TIMELINE */
        .tl { position: relative; padding-left: 28px; }
        .tl::before {
          content: ''; position: absolute; left: 0; top: 0; bottom: 0;
          width: 1px; background: var(--border);
        }
        .ti { position: relative; padding-bottom: 40px; }
        .ti::before {
          content: ''; position: absolute; left: -32px; top: 5px;
          width: 7px; height: 7px; border-radius: 50%;
          border: 1px solid var(--text-label); background: var(--bg);
        }
        .ti.done::before { background: var(--accent); border-color: var(--accent); }
        .ti-date {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; letter-spacing: 2px;
          text-transform: uppercase; color: var(--text-label); margin-bottom: 6px;
        }
        [data-theme="dark"] .ti-date { color: var(--accent); opacity: 0.7; }
        .ti-title { font-family: var(--serif); font-size: 20px; margin-bottom: 6px; }
        .ti-desc {
          font-size: 13px; color: var(--text-dim); line-height: 1.75;
          max-width: 440px; font-weight: 300;
        }

        /* PARTNERS */
        .ptg {
          display: flex; flex-wrap: wrap; gap: 1px;
          background: var(--border); border: 1px solid var(--border);
        }
        .pt {
          background: var(--bg); padding: 32px 28px;
          flex: 1; min-width: 230px; transition: background 0.3s;
        }
        [data-theme="dark"] .pt { background: var(--bg-card); }
        .pt:hover { background: var(--bg-card-hover); }
        .pt-role {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; letter-spacing: 2px;
          text-transform: uppercase; color: var(--text-label); margin-bottom: 6px;
        }
        [data-theme="dark"] .pt-role { color: var(--accent); opacity: 0.7; }
        .pt-name { font-family: var(--serif); font-size: 20px; margin-bottom: 3px; }
        .pt-desc { font-size: 13px; color: var(--text-dim); font-weight: 300; }

        .qs { padding: 80px 48px; max-width: 720px; margin: 0 auto; text-align: center; }
        .qs blockquote {
          font-family: var(--serif);
          font-size: clamp(20px, 3vw, 34px);
          line-height: 1.4; font-style: italic; font-weight: 400;
        }
        .qs-rule { width: 32px; height: 1px; background: var(--text-label); margin: 22px auto; }
        [data-theme="dark"] .qs-rule { background: var(--accent); opacity: 0.4; }
        .qs-attr {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; letter-spacing: 2px;
          color: var(--text-label); text-transform: uppercase;
        }

        .cta { padding: 100px 48px; text-align: center; }
        .cta-t {
          font-family: var(--serif);
          font-size: clamp(28px, 4.2vw, 50px);
          font-style: italic; font-weight: 400; margin-bottom: 16px;
        }
        [data-theme="dark"] .cta-t { color: var(--accent); }
        .cta-s {
          font-size: 15px; color: var(--text-dim); font-weight: 300;
          margin-bottom: 40px; max-width: 400px;
          margin-left: auto; margin-right: auto; line-height: 1.7;
        }

        .ft {
          border-top: 1px solid var(--border); padding: 32px 48px;
          display: flex; justify-content: space-between; align-items: center;
          max-width: 1060px; margin: 0 auto; flex-wrap: wrap; gap: 16px;
        }
        .ft-copy { font-size: 12px; color: var(--text-label); font-weight: 300; }
        .ft-links { display: flex; gap: 20px; list-style: none; }
        .ft-links a {
          color: var(--text-label); font-size: 12px; font-weight: 300; transition: color 0.3s;
        }
        .ft-links a:hover { color: var(--accent); }

        .nb { margin-top: 18px; padding: 24px; border: 1px solid var(--border); }
        [data-theme="dark"] .nb { background: var(--bg-card); border-color: var(--glass-border); }
        .nb-l {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px; letter-spacing: 2px;
          text-transform: uppercase; color: var(--text-label); margin-bottom: 10px;
        }
        [data-theme="dark"] .nb-l { color: var(--accent); opacity: 0.7; }
        .nb-t { font-size: 13px; color: var(--text-dim); line-height: 1.75; font-weight: 300; }
        .sk-grid {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
          gap: 14px; margin-top: 12px;
        }
        .sk-item { font-size: 13px; color: var(--text-dim); font-weight: 300; }

        @media (max-width: 768px) {
          .nav { padding: 14px 18px; }
          .nav-links { display: none; }
          .hero { padding: 96px 22px 56px; }
          .sec { padding: 64px 22px; }
          .pg { grid-template-columns: 1fr; }
          .hero-meta { gap: 22px; }
          .qs { padding: 56px 22px; }
          .cta { padding: 64px 22px; }
          .ft { padding: 24px 22px; }
        }
      `}</style>

        {/* NAV */}
        <nav className={`nav ${scrollY > 50 ? "scrolled" : ""}`}>
          <div className="nav-logo">DISTRIBUTE//AI</div>
          <div className="nav-right">
            <ul className="nav-links">
              {SECTIONS.map(s => <li key={s}><a onClick={() => go(s.toLowerCase())}>{s}</a></li>)}
            </ul>
            <button className="nav-btn" onClick={() => setDark(!dark)}>{dark ? "☀ Light" : "☾ Dark"}</button>
            <button className="nav-btn" onClick={() => go("register")}>Partner</button>
          </div>
        </nav>

        {/* HERO */}
        <section className="hero">
          <div className="hero-tag">Hackathon — April 2026 — London + Global</div>
          <h1>Decentralised<br /><em>AI Distribution.</em></h1>
          <p className="hero-sub">
            One collaborative build. Four specialisation tracks. Build a decentralised,
            encrypted, incentive-aligned LLM on blockchain — AI that belongs to everyone.
          </p>
          <div className="hero-meta">
            <div className="hero-meta-item">
              <span className="hero-meta-label">Hub</span>
              <span className="hero-meta-value">London</span>
            </div>
            <div className="hero-meta-item">
              <span className="hero-meta-label">Format</span>
              <span className="hero-meta-value">Global + Online</span>
            </div>
            <div className="hero-meta-item">
              <span className="hero-meta-label">Duration</span>
              <span className="hero-meta-value">48 Hours</span>
            </div>
            <div className="hero-meta-item">
              <span className="hero-meta-label">Tracks</span>
              <span className="hero-meta-value">4</span>
            </div>
          </div>
          <div className="hero-actions">
            <a href={`mailto:${CONTACT_EMAIL}`} className="hero-email">{CONTACT_EMAIL}</a>
            <button className="btn-o" onClick={() => go("about")}>Learn More</button>
          </div>
        </section>

        {/* QUOTE */}
        <Reveal>
          <section className="qs">
            <blockquote>
              "What if AI wasn't owned by three companies in San Francisco — but trained,
              encrypted, and served by anyone, anywhere, earning from their contribution?"
            </blockquote>
            <div className="qs-rule" />
            <span className="qs-attr">The Central Inquiry</span>
          </section>
        </Reveal>

        <div className="div" />

        {/* 01 — ABOUT */}
        <section className="sec" id="about">
          <Reveal>
            <div className="sec-num">01 — About</div>
            <h2 className="sec-title">One Project. Everyone Builds Together.</h2>
            <p className="sec-desc">
              This is not a competition between isolated teams. Everyone at DISTRIBUTE//AI
              works on the same project: building a decentralised LLM on blockchain — trained
              collaboratively, cryptographically secured, and economically open.
            </p>
          </Reveal>
          <Reveal delay={120}>
            <div className="cg">
              <div className="cd">
                <div className="cd-icon">∴</div>
                <h3>London Hub</h3>
                <p>The main venue. In-person hacking, speakers, mentors, and final presentations. The nerve centre of the build.</p>
              </div>
              <div className="cd">
                <div className="cd-icon">⊕</div>
                <h3>Online via Discord</h3>
                <p>Participate from anywhere. Real-time coordination, track channels, code review, and async collaboration across time zones.</p>
              </div>
              <div className="cd">
                <div className="cd-icon">⧉</div>
                <h3>Host Your City</h3>
                <p>Organise a satellite node in your city. We provide the branding, briefing packs, and coordination — you provide the space and the people.</p>
              </div>
            </div>
          </Reveal>
        </section>

        <div className="div" />

        {/* 02 — TRACKS */}
        <section className="sec" id="tracks">
          <Reveal>
            <div className="sec-num">02 — Tracks</div>
            <h2 className="sec-title">Four Tracks. One Architecture.</h2>
            <p className="sec-desc">
              Every participant joins a track. Every track feeds one system.
              The output is a single decentralised LLM — designed, encrypted,
              incentivised, and deployed collaboratively.
            </p>
          </Reveal>

          <Reveal delay={80}>
            <div className="arch">
              <div className="arch-label">System Architecture</div>
              <div className="arch-row">
                {["CS / ML", "Cryptography", "Mechanism Design", "Engineering"].map((t, i) => (
                  <div key={t} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <div className="arch-pill">{t}</div>
                    {i < 3 && <span className="arch-plus">+</span>}
                  </div>
                ))}
              </div>
              <div className="arch-arr">↓</div>
              <div className="arch-out">Decentralised LLM on Blockchain</div>
            </div>
          </Reveal>

          <Reveal delay={160}>
            <div className="cg" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))" }}>
              <div className="cd">
                <div className="cd-icon">∇</div>
                <h3>CS / ML</h3>
                <p>Design the LLM architecture for decentralised training. Model partitioning, federated learning, distributed gradient aggregation.</p>
                <div className="cd-tag">Track 01 — Model Design</div>
              </div>
              <div className="cd">
                <div className="cd-icon">◈</div>
                <h3>Cryptography</h3>
                <p>Encrypt model weights while preserving inference. Homomorphic encryption, secure multi-party computation, zero-knowledge proofs.</p>
                <div className="cd-tag">Track 02 — Weight Security</div>
              </div>
              <div className="cd">
                <div className="cd-icon">⊞</div>
                <h3>Mechanism Design</h3>
                <p>Design crypto-economic incentives: earn tokens by contributing training compute, and separately by providing distributed, anonymised inference.</p>
                <div className="cd-tag">Track 03 — Incentives</div>
              </div>
              <div className="cd">
                <div className="cd-icon">⬡</div>
                <h3>Engineering</h3>
                <p>Implementation and integration. Smart contracts, node infrastructure, API layer, deployment pipeline. Make the designs run.</p>
                <div className="cd-tag">Track 04 — Build</div>
              </div>
            </div>
          </Reveal>

          <Reveal delay={240}>
            <div className="nb">
              <div className="nb-l">How It Works</div>
              <p className="nb-t">
                Join the track that matches your skills. Each track has a lead and clear deliverables
                that plug into the shared architecture. Cross-track integration sessions happen every
                6 hours. The final demo is one system, presented together.
              </p>
            </div>
          </Reveal>
        </section>

        <div className="div" />

        {/* 03 — PRIZES */}
        <section className="sec" id="prizes">
          <Reveal>
            <div className="sec-num">03 — Rewards</div>
            <h2 className="sec-title">Prizes & Opportunities</h2>
            <p className="sec-desc">Cash prizes, compute credits, fellowships, and a platform to publish your work.</p>
          </Reveal>

          <Reveal delay={120}>
            <div className="pg">
              <div className="pr ft">
                <div className="pr-pl">1st Place</div>
                <div className="pr-am"><span className="c">$</span>2,000</div>
                <ul className="pr-pk">
                  <li>$500 compute credit</li>
                  <li>Bloomsbury Tech internship</li>
                  <li>Essay commission by Abelar</li>
                  <li>Sprint research community</li>
                </ul>
              </div>
              <div className="pr">
                <div className="pr-pl">2nd Place</div>
                <div className="pr-am"><span className="c">$</span>1,000</div>
                <ul className="pr-pk">
                  <li>Compute credits</li>
                  <li>Fellowship opportunity</li>
                  <li>Essay commission</li>
                </ul>
              </div>
              <div className="pr">
                <div className="pr-pl">3rd Place</div>
                <div className="pr-am"><span className="c">$</span>500</div>
                <ul className="pr-pk">
                  <li>Compute credits</li>
                  <li>Fellowship opportunity</li>
                  <li>Community access</li>
                </ul>
              </div>
            </div>
          </Reveal>

          <Reveal delay={200}>
            <div className="nb">
              <div className="nb-l">For All Participants</div>
              <p className="nb-t">
                Voluntary missions with Abelar — writing essays and policy memos on your hackathon
                work, commissioned and published. Join a sprint research community for continued
                exploration beyond the hackathon.
              </p>
            </div>
          </Reveal>
        </section>

        <div className="div" />

        {/* 04 — TIMELINE */}
        <section className="sec" id="timeline">
          <Reveal>
            <div className="sec-num">04 — Timeline</div>
            <h2 className="sec-title">From Concept to Impact.</h2>
          </Reveal>

          <Reveal delay={80}>
            <div className="tl">
              <div className="ti done">
                <div className="ti-date">March 2026</div>
                <div className="ti-title">Concept & Partnerships</div>
                <div className="ti-desc">Finalise concept note, secure funding and activation partners, draft partner deck, begin outreach.</div>
              </div>
              <div className="ti">
                <div className="ti-date">Early April 2026</div>
                <div className="ti-title">Launch & Registration</div>
                <div className="ti-desc">Marketing begins. Community partnerships activated. Participant registration opens. Discord server live.</div>
              </div>
              <div className="ti">
                <div className="ti-date">Mid April 2026</div>
                <div className="ti-title">Hackathon Weekend</div>
                <div className="ti-desc">48 hours. London main venue + satellite cities worldwide + Discord for global coordination. Cross-track integration every 6 hours. Final demo presented as one system.</div>
              </div>
              <div className="ti">
                <div className="ti-date">Post-Event</div>
                <div className="ti-title">Publication & Impact</div>
                <div className="ti-desc">Blog posts on Tech Policy Press. Impact report. Essay commissions. Sprint research community launched.</div>
              </div>
            </div>
          </Reveal>
        </section>

        <div className="div" />

        {/* 05 — PARTNERS */}
        <section className="sec" id="partners">
          <Reveal>
            <div className="sec-num">05 — Partners</div>
            <h2 className="sec-title">Who's Building This.</h2>
          </Reveal>

          <Reveal delay={120}>
            <div className="ptg">
              <div className="pt">
                <div className="pt-role">Technical Partner</div>
                <div className="pt-name">Bloomsbury Technology</div>
                <div className="pt-desc">AI research and ML engineering. Technical track design and mentorship.</div>
              </div>
              <div className="pt">
                <div className="pt-role">Impact Partner</div>
                <div className="pt-name">Abelar</div>
                <div className="pt-desc">Hackathon operations, impact analysis, essay commissioning, and voluntary missions.</div>
              </div>
            </div>
          </Reveal>

          <Reveal delay={200}>
            <div className="nb">
              <div className="nb-l">Seeking Partners</div>
              <div className="sk-grid">
                {["Funding Partners", "Compute Sponsors", "Crypto / Web3 Communities", "Cryptography Researchers", "City Node Organisers", "Judges & Speakers"].map(p => (
                  <div key={p} className="sk-item">— {p}</div>
                ))}
              </div>
            </div>
          </Reveal>
        </section>

        <div className="div" />

        {/* CTA */}
        <section className="cta" id="register">
          <h2 className="cta-t">Join the Build.</h2>
          <p className="cta-s">Hack in London. Join on Discord. Or host a node in your city. Pick your track. Ship one system.</p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
            <a href={`mailto:${CONTACT_EMAIL}`} className="hero-email">{CONTACT_EMAIL}</a>
          </div>
        </section>

        {/* FOOTER */}
        <div className="div" />
        <footer className="ft">
          <span className="ft-copy">© 2026 Bloomsbury Technology × Abelar</span>
          <ul className="ft-links">
            <li><a href="https://bloomsburytech.com" target="_blank" rel="noopener noreferrer">Bloomsbury Tech</a></li>
            <li><a href="#">GitHub</a></li>
            <li><a href="#">Discord</a></li>
            <li><a href={`mailto:${CONTACT_EMAIL}`}>Contact</a></li>
          </ul>
        </footer>
      </div>
  );
}
