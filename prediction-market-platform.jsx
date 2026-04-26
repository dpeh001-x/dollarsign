import { useState, useEffect, useCallback, useRef } from "react";

// ─── BACKEND CONFIG ───
const BACKEND_URL = (typeof window !== "undefined" && window.__ORACLE_BACKEND__) || "http://localhost:8000";

// ─── DATA GENERATION ENGINE (demo fallback only) ───
const CATEGORIES = ["Politics", "Crypto", "Economics", "Geopolitics", "Tech", "Sports", "Climate", "Culture"];
const SOURCES = ["Polymarket", "Kalshi", "Manifold", "PredictIt", "Metaculus", "Augur", "Betfair", "Smarkets"];
const SENTIMENTS = ["Bullish", "Bearish", "Neutral", "Conflicted"];

const MARKET_TEMPLATES = [
  { q: "Will US-Iran ceasefire hold past April 21?", cat: "Geopolitics", vol: 4200000, res: 2 },
  { q: "Oil above $110/bbl by end of April?", cat: "Economics", vol: 3800000, res: 8 },
  { q: "Fed rate cut before July 2026?", cat: "Economics", vol: 5100000, res: 72 },
  { q: "S&P 500 above 7000 by June 30?", cat: "Economics", vol: 6200000, res: 68 },
  { q: "Bitcoin above $100k by May 2026?", cat: "Crypto", vol: 8900000, res: 34 },
  { q: "Ethereum above $5000 by Q3 2026?", cat: "Crypto", vol: 3400000, res: 120 },
  { q: "Trump approval above 45% in May?", cat: "Politics", vol: 2100000, res: 30 },
  { q: "US recession declared by Q4 2026?", cat: "Economics", vol: 7200000, res: 180 },
  { q: "Strait of Hormuz fully reopened by May?", cat: "Geopolitics", vol: 4800000, res: 25 },
  { q: "China delivers weapons to Iran?", cat: "Geopolitics", vol: 2900000, res: 45 },
  { q: "NVIDIA earnings beat Q2 estimates?", cat: "Tech", vol: 5600000, res: 52 },
  { q: "Apple launches AI device in 2026?", cat: "Tech", vol: 1800000, res: 200 },
  { q: "Hezbollah ceasefire by May 2026?", cat: "Geopolitics", vol: 3100000, res: 35 },
  { q: "VIX stays above 25 through April?", cat: "Economics", vol: 4400000, res: 6 },
  { q: "Gold above $3500/oz by June?", cat: "Economics", vol: 3700000, res: 60 },
  { q: "XOM hits new ATH by May 15?", cat: "Economics", vol: 2600000, res: 20 },
  { q: "RTX stock above $230 by June?", cat: "Economics", vol: 1900000, res: 55 },
  { q: "Ukraine ceasefire deal in 2026?", cat: "Geopolitics", vol: 4100000, res: 250 },
  { q: "Next Pope visits war zone in 2026?", cat: "Culture", vol: 800000, res: 180 },
  { q: "US GDP growth above 2% in Q2?", cat: "Economics", vol: 3300000, res: 75 },
  { q: "Iran nuclear deal signed by July?", cat: "Geopolitics", vol: 5500000, res: 90 },
  { q: "Solana above $300 by Q3?", cat: "Crypto", vol: 2200000, res: 100 },
  { q: "Tesla Robotaxi launch in 2026?", cat: "Tech", vol: 4000000, res: 220 },
  { q: "US unemployment above 5% by Dec?", cat: "Economics", vol: 2800000, res: 240 },
  { q: "Major cyberattack on US grid 2026?", cat: "Tech", vol: 1500000, res: 260 },
  { q: "Democrats win House in midterms?", cat: "Politics", vol: 6800000, res: 190 },
  { q: "Oil below $80 by year end?", cat: "Economics", vol: 3600000, res: 250 },
  { q: "Lebanon-Israel peace treaty 2026?", cat: "Geopolitics", vol: 2400000, res: 180 },
  { q: "CPI below 3% by September?", cat: "Economics", vol: 4700000, res: 140 },
  { q: "Alibaba stock above $180 by Dec?", cat: "Economics", vol: 2100000, res: 250 },
];

function generateMarkets() {
  return MARKET_TEMPLATES.map((t, i) => {
    const price = Math.round((15 + Math.random() * 70) * 100) / 100;
    const spread = Math.round((0.5 + Math.random() * 8) * 100) / 100;
    const priceMove24h = Math.round((Math.random() * 20 - 10) * 100) / 100;
    const vol = Math.round(t.vol * (0.7 + Math.random() * 0.6));
    const isAnomalous = Math.abs(priceMove24h) > 7 || spread > 5;
    const sentScore = Math.round((Math.random() * 200 - 100)) / 100;
    const modelProb = Math.round((price + (Math.random() * 24 - 12)) * 100) / 100;
    const edge = Math.round((modelProb - price) * 100) / 100;
    const confidence = Math.round((50 + Math.random() * 50) * 10) / 10;
    return {
      id: i,
      question: t.q,
      category: t.cat,
      source: SOURCES[i % SOURCES.length],
      marketPrice: Math.min(95, Math.max(5, price)),
      spread,
      volume: vol,
      daysToResolution: t.res,
      priceMove24h,
      isAnomalous,
      sentimentScore: sentScore,
      sentimentLabel: sentScore > 0.5 ? "Bullish" : sentScore < -0.5 ? "Bearish" : Math.abs(sentScore) < 0.2 ? "Neutral" : "Conflicted",
      modelProb: Math.min(95, Math.max(5, modelProb)),
      edge: Math.abs(edge) > 2 ? edge : 0,
      confidence,
      hasTrade: Math.abs(edge) > 3 && confidence > 72,
      twitterMentions: Math.floor(Math.random() * 12000),
      redditPosts: Math.floor(Math.random() * 800),
      newsArticles: Math.floor(Math.random() * 50),
    };
  });
}

// ─── LIVE BACKEND FETCH ───
function timeAgo(unixSecs) {
  if (!unixSecs) return "—";
  const diff = Math.max(0, Math.floor(Date.now() / 1000 - unixSecs));
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

async function fetchLiveMarkets() {
  const res = await fetch(`${BACKEND_URL}/api/markets?limit=100`);
  if (!res.ok) throw new Error(`Backend ${res.status}`);
  const data = await res.json();
  const markets = (data.markets || []).map((m, i) => {
    const price = Math.min(99.5, Math.max(0.5, Number(m.marketPrice) || 50));
    return {
      id: m.id || `${m.source}-${i}`,
      question: m.question || "Untitled market",
      category: m.category || "Other",
      source: m.source,
      marketPrice: price,
      spread: Number(m.spread) || 0,
      volume: Number(m.volume || m.totalVolume || 0),
      daysToResolution: m.daysToResolution ?? 30,
      priceMove24h: Number(m.priceMove24h) || 0,
      isAnomalous: (Number(m.spread) || 0) > 5 || Math.abs(Number(m.priceMove24h) || 0) > 7,
      sentimentScore: 0,
      sentimentLabel: "Pending",
      modelProb: price,
      edge: 0,
      confidence: 0,
      hasTrade: false,
      twitterMentions: 0,
      redditPosts: 0,
      newsArticles: 0,
      slug: m.slug,
      ticker: m.ticker,
    };
  });
  return { markets, counts: data.counts || {}, errors: data.errors || {} };
}

async function fetchLiveSentiment() {
  const res = await fetch(`${BACKEND_URL}/api/sentiment?limit=20`);
  if (!res.ok) throw new Error(`Backend ${res.status}`);
  const data = await res.json();
  const feed = (data.feed || []).map(p => ({
    src: p.src || "Reddit",
    user: p.user || "",
    text: p.text || "",
    sent: Number(p.sent) || 0,
    time: timeAgo(p.createdUtc),
    url: p.url,
    score: p.score,
  }));
  return { feed, aggregate: data.aggregate ?? 0, configured: data.configured };
}

function generateSentimentFeed() {
  const items = [
    { src: "X / Twitter", user: "@zerohedge", text: "Strait of Hormuz closure = $130 oil incoming. This is NOT priced in.", sent: -0.82, time: "2m ago" },
    { src: "Reddit", user: "r/wallstreetbets", text: "XOM calls printing money while everyone panic sells tech 🛢️🔥", sent: 0.71, time: "5m ago" },
    { src: "Truth Social", user: "@realDonaldTrump", text: "Deal with Iran will happen. They have no choice. We are winning BIG!", sent: 0.45, time: "12m ago" },
    { src: "X / Twitter", user: "@babortrading", text: "VIX term structure inverted again. Ceasefire expiry Wednesday is the trigger.", sent: -0.65, time: "18m ago" },
    { src: "RSS / Reuters", user: "Reuters Breaking", text: "Iran IRGC warns of 'decisive response' after US seizes cargo vessel in Gulf of Oman", sent: -0.91, time: "22m ago" },
    { src: "Reddit", user: "r/geopolitics", text: "Analysis: Ghalibaf framing Hormuz as victory means Iran won't give it up cheaply", sent: -0.38, time: "31m ago" },
    { src: "X / Twitter", user: "@GoldmanSachs", text: "Raising oil price target to $120 on escalation risk. Energy overweight maintained.", sent: 0.55, time: "45m ago" },
    { src: "RSS / Al Jazeera", user: "Al Jazeera Live", text: "23 ships turned around since US blockade began April 13. Global supply chains disrupted.", sent: -0.77, time: "1h ago" },
    { src: "X / Twitter", user: "@NateSilver", text: "Prediction markets pricing ceasefire holding at 35%. That feels about right given ship seizure.", sent: -0.2, time: "1h ago" },
    { src: "Reddit", user: "r/options", text: "Loaded up on QQQ puts expiring Friday. If ceasefire collapses Wednesday, blood bath.", sent: -0.85, time: "2h ago" },
  ];
  return items;
}

const TRADE_HISTORY = [
  { id: 1, market: "US-Iran ceasefire extends past April 8", side: "NO", entry: 62, exit: 78, pnl: -320, size: 2000, date: "Apr 7", result: "loss", postMortem: "Entered too early before peace talks details were known. Ceasefire did initially extend. Lesson: wait for catalyst confirmation before fading consensus." },
  { id: 2, market: "Oil above $100 by April 14", side: "YES", entry: 55, exit: 82, pnl: 540, size: 2000, date: "Apr 10", result: "win", postMortem: null },
  { id: 3, market: "VIX spike above 30 during talks", side: "YES", entry: 40, exit: 22, pnl: -360, size: 2000, date: "Apr 11", result: "loss", postMortem: "Overestimated panic response to failed talks. VIX spiked to 28 but not 30. Model was 68% confident — below threshold. Shouldn't have taken the trade. Lesson: respect the 72% confidence threshold." },
  { id: 4, market: "XOM hits $170 by April 15", side: "YES", entry: 45, exit: 71, pnl: 520, size: 2000, date: "Apr 12", result: "win", postMortem: null },
  { id: 5, market: "Bitcoin drops below $70k on war news", side: "YES", entry: 35, exit: 18, pnl: -340, size: 2000, date: "Apr 14", result: "loss", postMortem: "Bitcoin showed flight-to-safety behavior instead of risk-off. Sentiment analysis was bearish but crypto narrative decoupled from trad-fi. Lesson: crypto correlations break during geopolitical events — reduce position size by 50% on crypto-geo plays." },
];

// ─── STYLES ───
const fonts = `@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');`;

const C = {
  bg: "#0A0B0E",
  surface: "#12131A",
  surface2: "#1A1B24",
  surface3: "#22232E",
  border: "#2A2B38",
  border2: "#3A3B4A",
  text: "#E8E6E1",
  text2: "#9B99A1",
  text3: "#66656E",
  green: "#00E68A",
  greenDim: "#00E68A22",
  red: "#FF4757",
  redDim: "#FF475722",
  orange: "#FF9F43",
  orangeDim: "#FF9F4322",
  blue: "#4FACFE",
  blueDim: "#4FACFE22",
  purple: "#A855F7",
  purpleDim: "#A855F722",
  yellow: "#FFD93D",
};

// ─── COMPONENTS ───
const Badge = ({ color, children }) => (
  <span style={{ fontSize: 10, fontWeight: 600, padding: "2px 8px", borderRadius: 4, background: color + "22", color, fontFamily: "'JetBrains Mono', monospace", letterSpacing: 0.5, textTransform: "uppercase" }}>{children}</span>
);

const Stat = ({ label, value, color, small }) => (
  <div style={{ background: C.surface2, borderRadius: 8, padding: small ? "8px 10px" : "10px 14px", border: `1px solid ${C.border}` }}>
    <div style={{ fontSize: 10, color: C.text3, marginBottom: 2, fontFamily: "'JetBrains Mono', monospace", letterSpacing: 0.8, textTransform: "uppercase" }}>{label}</div>
    <div style={{ fontSize: small ? 15 : 18, fontWeight: 700, color: color || C.text, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
  </div>
);

const ProgressBar = ({ value, max, color, height = 4 }) => (
  <div style={{ width: "100%", height, background: C.surface3, borderRadius: height / 2, overflow: "hidden" }}>
    <div style={{ width: `${Math.min((value / max) * 100, 100)}%`, height: "100%", background: color, borderRadius: height / 2, transition: "width 0.6s ease" }} />
  </div>
);

const Tab = ({ active, children, onClick }) => (
  <button onClick={onClick} style={{ background: active ? C.surface3 : "transparent", border: `1px solid ${active ? C.border2 : "transparent"}`, borderRadius: 6, padding: "7px 16px", fontSize: 12, fontWeight: active ? 600 : 400, color: active ? C.text : C.text3, cursor: "pointer", fontFamily: "'Outfit', sans-serif", transition: "all 0.2s", whiteSpace: "nowrap" }}>{children}</button>
);

// ─── MAIN APP ───
export default function PredictionPlatform() {
  const [tab, setTab] = useState("scanner");
  const [markets, setMarkets] = useState([]);
  const [feed, setFeed] = useState([]);
  const [bankroll, setBankroll] = useState(10000);
  const [bankrollInput, setBankrollInput] = useState("10000");
  const [kellyFraction, setKellyFraction] = useState(0.25);
  const [confidenceThreshold, setConfidenceThreshold] = useState(72);
  const [sortBy, setSortBy] = useState("volume");
  const [filterCat, setFilterCat] = useState("All");
  const [filterAnomaly, setFilterAnomaly] = useState(false);
  const [filterTrades, setFilterTrades] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanComplete, setScanComplete] = useState(false);
  const [expandedTrade, setExpandedTrade] = useState(null);
  const [liveMode, setLiveMode] = useState(false);
  const [backendStatus, setBackendStatus] = useState({ counts: {}, errors: {}, sentimentConfigured: false, error: null });
  const scanInterval = useRef(null);

  const runScan = useCallback(async () => {
    setScanning(true);
    setScanProgress(0);
    setScanComplete(false);

    let p = 0;
    scanInterval.current = setInterval(() => {
      p += Math.random() * 6 + 2;
      setScanProgress(Math.min(p, 95));
    }, 120);

    try {
      const [marketRes, sentRes] = await Promise.allSettled([
        fetchLiveMarkets(),
        fetchLiveSentiment(),
      ]);

      if (marketRes.status === "fulfilled" && marketRes.value.markets.length > 0) {
        setMarkets(marketRes.value.markets);
        const sentimentConfigured = sentRes.status === "fulfilled" && sentRes.value.configured;
        setFeed(
          sentRes.status === "fulfilled" && sentRes.value.feed.length > 0
            ? sentRes.value.feed
            : generateSentimentFeed()
        );
        setLiveMode(true);
        setBackendStatus({
          counts: marketRes.value.counts,
          errors: marketRes.value.errors,
          sentimentConfigured,
          error: null,
        });
      } else {
        const err = marketRes.status === "rejected" ? marketRes.reason?.message : "no markets returned";
        setMarkets(generateMarkets());
        setFeed(generateSentimentFeed());
        setLiveMode(false);
        setBackendStatus({ counts: {}, errors: {}, sentimentConfigured: false, error: err });
      }
    } catch (e) {
      setMarkets(generateMarkets());
      setFeed(generateSentimentFeed());
      setLiveMode(false);
      setBackendStatus({ counts: {}, errors: {}, sentimentConfigured: false, error: e.message });
    } finally {
      clearInterval(scanInterval.current);
      setScanProgress(100);
      setScanning(false);
      setScanComplete(true);
    }
  }, []);

  useEffect(() => { runScan(); return () => clearInterval(scanInterval.current); }, []);

  const sortedMarkets = [...markets]
    .filter(m => filterCat === "All" || m.category === filterCat)
    .filter(m => !filterAnomaly || m.isAnomalous)
    .filter(m => !filterTrades || m.hasTrade)
    .sort((a, b) => {
      if (sortBy === "volume") return b.volume - a.volume;
      if (sortBy === "edge") return Math.abs(b.edge) - Math.abs(a.edge);
      if (sortBy === "confidence") return b.confidence - a.confidence;
      if (sortBy === "resolution") return a.daysToResolution - b.daysToResolution;
      if (sortBy === "spread") return b.spread - a.spread;
      return 0;
    });

  const anomalyCount = markets.filter(m => m.isAnomalous).length;
  const tradeCount = markets.filter(m => m.hasTrade).length;
  const avgEdge = markets.filter(m => m.hasTrade).reduce((a, m) => a + Math.abs(m.edge), 0) / (tradeCount || 1);
  const totalPnl = TRADE_HISTORY.reduce((a, t) => a + t.pnl, 0);
  const winRate = Math.round((TRADE_HISTORY.filter(t => t.result === "win").length / TRADE_HISTORY.length) * 100);

  const applyBankroll = () => {
    const v = parseFloat(bankrollInput);
    if (!isNaN(v) && v > 0) setBankroll(v);
  };

  return (
    <div style={{ fontFamily: "'Outfit', sans-serif", background: C.bg, color: C.text, minHeight: "100vh", padding: "0" }}>
      <style>{fonts}</style>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: ${C.surface}; }
        ::-webkit-scrollbar-thumb { background: ${C.border2}; border-radius: 2px; }
        input[type=range] { -webkit-appearance: none; height: 4px; background: ${C.surface3}; border-radius: 2px; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: ${C.blue}; cursor: pointer; }
      `}</style>

      {/* Header */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: "12px 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: scanComplete ? C.green : C.orange, boxShadow: `0 0 8px ${scanComplete ? C.green : C.orange}60`, animation: scanning ? "pulse 1s infinite" : "none" }} />
            <style>{`@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }`}</style>
            <span style={{ fontSize: 16, fontWeight: 700, letterSpacing: -0.5 }}>ORACLE</span>
            <span style={{ fontSize: 11, color: C.text3, fontFamily: "'JetBrains Mono', monospace" }}>prediction intelligence</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 10, fontWeight: 600, padding: "3px 8px", borderRadius: 4, background: liveMode ? C.greenDim : C.orangeDim, color: liveMode ? C.green : C.orange, fontFamily: "'JetBrains Mono', monospace", letterSpacing: 0.5 }}>
              {liveMode ? "● LIVE" : "○ DEMO"}
            </span>
            <span style={{ fontSize: 11, color: C.text3, fontFamily: "'JetBrains Mono', monospace" }}>{scanComplete ? `${markets.length} markets loaded` : scanning ? `scanning... ${Math.round(scanProgress)}%` : "ready"}</span>
            <button onClick={runScan} disabled={scanning} style={{ background: C.blue, border: "none", borderRadius: 6, padding: "6px 14px", fontSize: 11, fontWeight: 600, color: "#000", cursor: scanning ? "wait" : "pointer", opacity: scanning ? 0.5 : 1, fontFamily: "'Outfit', sans-serif" }}>
              {scanning ? "Scanning..." : "⟳ Rescan"}
            </button>
          </div>
        </div>
        {scanning && (
          <div style={{ marginTop: 8 }}>
            <ProgressBar value={scanProgress} max={100} color={C.blue} height={3} />
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
              <span style={{ fontSize: 10, color: C.text3, fontFamily: "'JetBrains Mono', monospace" }}>
                {scanProgress < 20 ? "Scanning markets..." : scanProgress < 40 ? "Filtering by liquidity & volume..." : scanProgress < 60 ? "Scraping social feeds..." : scanProgress < 80 ? "Running sentiment analysis..." : scanProgress < 95 ? "XGBoost + LLM calibration..." : "Finalizing predictions..."}
              </span>
              <span style={{ fontSize: 10, color: C.text3, fontFamily: "'JetBrains Mono', monospace" }}>{Math.round(scanProgress)}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Live/demo status banner */}
      {scanComplete && (
        <div style={{ padding: "8px 20px", background: liveMode ? C.greenDim : C.orangeDim, borderBottom: `1px solid ${liveMode ? C.green + "33" : C.orange + "33"}`, fontSize: 11, color: liveMode ? C.green : C.orange, fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.5 }}>
          {liveMode ? (
            <>
              LIVE · Polymarket: {backendStatus.counts.polymarket || 0} · Kalshi: {backendStatus.counts.kalshi || 0}
              {!backendStatus.sentimentConfigured && " · Reddit: not configured (set REDDIT_CLIENT_ID/SECRET in backend/.env)"}
              {Object.keys(backendStatus.errors).length > 0 && ` · errors: ${Object.entries(backendStatus.errors).map(([k, v]) => `${k}=${v}`).join("; ")}`}
              {" · "}
              <span style={{ color: C.text3 }}>XGBoost not yet trained — model probabilities default to market price (no edge until training pass runs)</span>
            </>
          ) : (
            <>DEMO · Backend at {BACKEND_URL} unreachable{backendStatus.error ? ` (${backendStatus.error})` : ""} — start it with: cd backend &amp;&amp; uvicorn main:app --reload</>
          )}
        </div>
      )}

      {/* Stats bar */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))", gap: 8, padding: "12px 20px", background: C.surface, borderBottom: `1px solid ${C.border}` }}>
        <Stat label="Markets scanned" value={markets.length} small />
        <Stat label="Anomalies" value={anomalyCount} color={anomalyCount > 5 ? C.red : C.orange} small />
        <Stat label="Live trades" value={tradeCount} color={C.green} small />
        <Stat label="Avg edge" value={`${avgEdge.toFixed(1)}¢`} color={C.blue} small />
        <Stat label="Session P&L" value={`$${totalPnl > 0 ? "+" : ""}${totalPnl}`} color={totalPnl >= 0 ? C.green : C.red} small />
        <Stat label="Win rate" value={`${winRate}%`} color={winRate >= 50 ? C.green : C.red} small />
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, padding: "10px 20px", borderBottom: `1px solid ${C.border}`, overflowX: "auto" }}>
        {[
          ["scanner", "Market Scanner"],
          ["sentiment", "Sentiment Feed"],
          ["predictions", "Prediction Agent"],
          ["risk", "Risk Agent"],
          ["history", "Trade History"],
        ].map(([key, label]) => (
          <Tab key={key} active={tab === key} onClick={() => setTab(key)}>{label}</Tab>
        ))}
      </div>

      <div style={{ padding: "16px 20px" }}>
        {/* ─── SCANNER TAB ─── */}
        {tab === "scanner" && (
          <div>
            {/* Filters */}
            <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
              <select value={filterCat} onChange={e => setFilterCat(e.target.value)} style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 6, padding: "6px 10px", fontSize: 12, color: C.text, fontFamily: "'Outfit', sans-serif" }}>
                <option value="All">All categories</option>
                {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
              <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 6, padding: "6px 10px", fontSize: 12, color: C.text, fontFamily: "'Outfit', sans-serif" }}>
                <option value="volume">Sort: Volume</option>
                <option value="edge">Sort: Edge</option>
                <option value="confidence">Sort: Confidence</option>
                <option value="resolution">Sort: Time to resolve</option>
                <option value="spread">Sort: Spread</option>
              </select>
              <button onClick={() => setFilterAnomaly(!filterAnomaly)} style={{ background: filterAnomaly ? C.redDim : C.surface2, border: `1px solid ${filterAnomaly ? C.red + "44" : C.border}`, borderRadius: 6, padding: "6px 12px", fontSize: 11, color: filterAnomaly ? C.red : C.text3, cursor: "pointer", fontFamily: "'Outfit', sans-serif" }}>
                ⚠ Anomalies ({anomalyCount})
              </button>
              <button onClick={() => setFilterTrades(!filterTrades)} style={{ background: filterTrades ? C.greenDim : C.surface2, border: `1px solid ${filterTrades ? C.green + "44" : C.border}`, borderRadius: 6, padding: "6px 12px", fontSize: 11, color: filterTrades ? C.green : C.text3, cursor: "pointer", fontFamily: "'Outfit', sans-serif" }}>
                ✦ Tradeable ({tradeCount})
              </button>
            </div>

            {/* Market table */}
            <div style={{ overflowX: "auto" }}>
              <div style={{ minWidth: 900 }}>
                {/* Header */}
                <div style={{ display: "grid", gridTemplateColumns: "2.5fr 0.7fr 0.7fr 0.8fr 0.6fr 0.7fr 0.7fr 0.6fr 0.6fr", gap: 8, padding: "8px 12px", fontSize: 10, color: C.text3, fontFamily: "'JetBrains Mono', monospace", letterSpacing: 0.5, textTransform: "uppercase", borderBottom: `1px solid ${C.border}` }}>
                  <span>Market</span><span>Price</span><span>Spread</span><span>Volume</span><span>Resolve</span><span>24h Δ</span><span>Sentiment</span><span>Model</span><span>Edge</span>
                </div>
                {/* Rows */}
                {sortedMarkets.map(m => (
                  <div key={m.id} style={{ display: "grid", gridTemplateColumns: "2.5fr 0.7fr 0.7fr 0.8fr 0.6fr 0.7fr 0.7fr 0.6fr 0.6fr", gap: 8, padding: "10px 12px", fontSize: 12, borderBottom: `1px solid ${C.border}`, background: m.isAnomalous ? C.redDim : m.hasTrade ? C.greenDim : "transparent", alignItems: "center", transition: "background 0.2s" }}>
                    <div>
                      <div style={{ fontWeight: 500, fontSize: 12, lineHeight: 1.3, marginBottom: 3 }}>{m.question}</div>
                      <div style={{ display: "flex", gap: 4 }}>
                        <Badge color={C.text3}>{m.source}</Badge>
                        <Badge color={C.purple}>{m.category}</Badge>
                        {m.isAnomalous && <Badge color={C.red}>ANOMALY</Badge>}
                        {m.hasTrade && <Badge color={C.green}>TRADE</Badge>}
                      </div>
                    </div>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{m.marketPrice.toFixed(1)}¢</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", color: m.spread > 5 ? C.red : m.spread > 3 ? C.orange : C.text2 }}>{m.spread.toFixed(1)}¢</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", color: C.text2 }}>${(m.volume / 1e6).toFixed(1)}M</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", color: m.daysToResolution <= 7 ? C.orange : C.text2 }}>{m.daysToResolution}d</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: m.priceMove24h > 0 ? C.green : m.priceMove24h < 0 ? C.red : C.text3 }}>{m.priceMove24h > 0 ? "+" : ""}{m.priceMove24h.toFixed(1)}¢</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: m.sentimentLabel === "Bullish" ? C.green : m.sentimentLabel === "Bearish" ? C.red : C.text3 }}>{m.sentimentLabel}</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: C.blue }}>{m.modelProb.toFixed(1)}¢</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: m.edge > 0 ? C.green : m.edge < 0 ? C.red : C.text3 }}>{m.edge > 0 ? "+" : ""}{m.edge.toFixed(1)}¢</span>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ fontSize: 10, color: C.text3, marginTop: 8, fontFamily: "'JetBrains Mono', monospace" }}>
              Showing {sortedMarkets.length} of {markets.length} markets · Anomaly = |24h move| &gt; 7¢ or spread &gt; 5¢ · Trade = |edge| &gt; 3¢ and confidence &gt; {confidenceThreshold}%
            </div>
          </div>
        )}

        {/* ─── SENTIMENT TAB ─── */}
        {tab === "sentiment" && (
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Live Social & News Sentiment Feed</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {feed.map((item, i) => (
                <div key={i} style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 14px", borderLeft: `3px solid ${item.sent > 0.3 ? C.green : item.sent < -0.3 ? C.red : C.text3}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      <Badge color={item.src.includes("Twitter") || item.src.includes("X") ? C.blue : item.src.includes("Reddit") ? C.orange : item.src.includes("Truth") ? C.purple : C.text3}>{item.src}</Badge>
                      <span style={{ fontSize: 12, fontWeight: 600, color: C.text }}>{item.user}</span>
                    </div>
                    <span style={{ fontSize: 10, color: C.text3, fontFamily: "'JetBrains Mono', monospace" }}>{item.time}</span>
                  </div>
                  <div style={{ fontSize: 13, color: C.text, lineHeight: 1.5, marginBottom: 6 }}>{item.text}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 10, color: C.text3, fontFamily: "'JetBrains Mono', monospace" }}>SENTIMENT</span>
                    <div style={{ width: 120, height: 6, background: C.surface3, borderRadius: 3, overflow: "hidden", position: "relative" }}>
                      <div style={{ position: "absolute", left: "50%", width: `${Math.abs(item.sent) * 50}%`, height: "100%", background: item.sent > 0 ? C.green : C.red, borderRadius: 3, ...(item.sent < 0 ? { right: "50%", left: "auto" } : {}) }} />
                    </div>
                    <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: item.sent > 0 ? C.green : C.red }}>{item.sent > 0 ? "+" : ""}{item.sent.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
            <div style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, padding: 14, marginTop: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Aggregate Sentiment Summary</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8 }}>
                <Stat label="X / Twitter" value="-0.42" color={C.red} small />
                <Stat label="Reddit" value="+0.18" color={C.green} small />
                <Stat label="Truth Social" value="+0.45" color={C.green} small />
                <Stat label="RSS / News" value="-0.61" color={C.red} small />
              </div>
              <div style={{ fontSize: 11, color: C.text3, marginTop: 8, lineHeight: 1.6 }}>
                <span style={{ color: C.red, fontWeight: 600 }}>Net bearish (-0.35).</span> Narrative divergence detected: social media is split (retail bullish on energy, bearish on tech) while news feeds are uniformly negative on geopolitical resolution. This divergence vs market odds (ceasefire at 35%) suggests market is pricing news correctly but underweighting social momentum.
              </div>
            </div>
          </div>
        )}

        {/* ─── PREDICTION AGENT TAB ─── */}
        {tab === "predictions" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
              <div style={{ fontSize: 13, fontWeight: 600 }}>XGBoost + LLM Prediction Agent</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 11, color: C.text3 }}>Confidence threshold:</span>
                <input type="range" min="50" max="95" value={confidenceThreshold} onChange={e => setConfidenceThreshold(Number(e.target.value))} style={{ width: 100 }} />
                <span style={{ fontSize: 12, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: C.blue }}>{confidenceThreshold}%</span>
              </div>
            </div>

            <div style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, padding: 14, marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: C.text3, fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>MODEL PIPELINE</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 6 }}>
                {["Feature extraction", "XGBoost ensemble", "LLM calibration", "Edge calculation", "Confidence filter"].map((step, i) => (
                  <div key={i} style={{ background: C.surface3, borderRadius: 6, padding: "8px 10px", textAlign: "center" }}>
                    <div style={{ fontSize: 10, color: C.text3, marginBottom: 2 }}>STEP {i + 1}</div>
                    <div style={{ fontSize: 11, fontWeight: 500 }}>{step}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 10, color: C.text3, marginTop: 8, lineHeight: 1.5 }}>
                Features: market price, volume, spread, 24h momentum, days to resolution, category, sentiment scores (X, Reddit, Truth Social, RSS), news article count, historical accuracy by source, VIX, oil price, S&P 500 level. XGBoost outputs raw probability → LLM adjusts for narrative context and known biases → final calibrated probability.
              </div>
            </div>

            {/* Trade signals */}
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Active Trade Signals ({markets.filter(m => m.hasTrade && m.confidence >= confidenceThreshold).length})</div>
            {markets.filter(m => m.hasTrade && m.confidence >= confidenceThreshold).length === 0 && (
              <div style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20, textAlign: "center", color: C.text3 }}>
                No signals above {confidenceThreshold}% confidence threshold. Lower the threshold or wait for new market data.
              </div>
            )}
            {markets.filter(m => m.hasTrade && m.confidence >= confidenceThreshold).sort((a, b) => b.confidence - a.confidence).map(m => {
              const side = m.edge > 0 ? "YES" : "NO";
              const kellyBet = Math.round(bankroll * kellyFraction * (Math.abs(m.edge) / 100) * (m.confidence / 100));
              return (
                <div key={m.id} style={{ background: C.surface2, border: `1px solid ${side === "YES" ? C.green + "33" : C.red + "33"}`, borderRadius: 8, padding: 14, marginBottom: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{m.question}</div>
                      <div style={{ display: "flex", gap: 4 }}><Badge color={C.purple}>{m.category}</Badge><Badge color={C.text3}>{m.source}</Badge></div>
                    </div>
                    <Badge color={side === "YES" ? C.green : C.red}>BUY {side} @ {m.marketPrice.toFixed(1)}¢</Badge>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(110px, 1fr))", gap: 6, marginBottom: 8 }}>
                    <Stat label="Market" value={`${m.marketPrice.toFixed(1)}¢`} small />
                    <Stat label="Model prob" value={`${m.modelProb.toFixed(1)}¢`} color={C.blue} small />
                    <Stat label="Edge" value={`${m.edge > 0 ? "+" : ""}${m.edge.toFixed(1)}¢`} color={m.edge > 0 ? C.green : C.red} small />
                    <Stat label="Confidence" value={`${m.confidence.toFixed(0)}%`} color={m.confidence > 80 ? C.green : C.orange} small />
                    <Stat label="Kelly bet" value={`$${kellyBet}`} color={C.yellow} small />
                    <Stat label="Resolves" value={`${m.daysToResolution}d`} small />
                  </div>
                  <ProgressBar value={m.confidence} max={100} color={m.confidence > 80 ? C.green : C.orange} height={3} />
                </div>
              );
            })}
          </div>
        )}

        {/* ─── RISK AGENT TAB ─── */}
        {tab === "risk" && (
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Risk Management Agent</div>

            {/* Bankroll input */}
            <div style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, padding: 14, marginBottom: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10 }}>Account Configuration</div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 12, color: C.text3 }}>Bankroll ($):</span>
                  <input type="number" value={bankrollInput} onChange={e => setBankrollInput(e.target.value)} style={{ background: C.surface3, border: `1px solid ${C.border}`, borderRadius: 6, padding: "6px 10px", fontSize: 14, fontWeight: 600, color: C.text, fontFamily: "'JetBrains Mono', monospace", width: 120 }} />
                </div>
                <button onClick={applyBankroll} style={{ background: C.green, border: "none", borderRadius: 6, padding: "6px 14px", fontSize: 12, fontWeight: 600, color: "#000", cursor: "pointer" }}>Apply</button>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 12 }}>
                  <span style={{ fontSize: 12, color: C.text3 }}>Kelly fraction:</span>
                  <input type="range" min="0.05" max="1" step="0.05" value={kellyFraction} onChange={e => setKellyFraction(Number(e.target.value))} style={{ width: 100 }} />
                  <span style={{ fontSize: 12, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: C.blue }}>{(kellyFraction * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>

            {/* Risk metrics */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 8, marginBottom: 12 }}>
              <Stat label="Total bankroll" value={`$${bankroll.toLocaleString()}`} color={C.text} />
              <Stat label="Max single bet" value={`$${Math.round(bankroll * 0.05).toLocaleString()}`} color={C.orange} />
              <Stat label="Max total exposure" value={`$${Math.round(bankroll * 0.20).toLocaleString()}`} color={C.yellow} />
              <Stat label="Current exposure" value={`$${Math.round(bankroll * 0.08).toLocaleString()}`} color={C.green} />
              <Stat label="Daily loss limit" value={`$${Math.round(bankroll * 0.03).toLocaleString()}`} color={C.red} />
              <Stat label="Session P&L" value={`$${totalPnl > 0 ? "+" : ""}${totalPnl}`} color={totalPnl >= 0 ? C.green : C.red} />
            </div>

            {/* Position sizing for active signals */}
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Position Sizing for Active Signals</div>
            {markets.filter(m => m.hasTrade && m.confidence >= confidenceThreshold).map(m => {
              const kellyFull = (m.confidence / 100) - ((1 - m.confidence / 100) / ((100 / m.marketPrice) - 1));
              const kellyAdj = Math.max(0, kellyFull * kellyFraction);
              const betSize = Math.round(bankroll * kellyAdj);
              const maxBet = bankroll * 0.05;
              const capped = Math.min(betSize, maxBet);
              const isCapped = betSize > maxBet;
              const riskPct = ((capped / bankroll) * 100).toFixed(1);

              return (
                <div key={m.id} style={{ background: C.surface2, border: `1px solid ${isCapped ? C.orange + "44" : C.border}`, borderRadius: 8, padding: 12, marginBottom: 6 }}>
                  <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 6 }}>{m.question}</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(100px, 1fr))", gap: 6 }}>
                    <Stat label="Kelly raw" value={`$${betSize}`} small />
                    <Stat label="Capped bet" value={`$${capped}`} color={isCapped ? C.orange : C.green} small />
                    <Stat label="% of bankroll" value={`${riskPct}%`} color={parseFloat(riskPct) > 4 ? C.orange : C.green} small />
                    <Stat label="Max loss" value={`-$${capped}`} color={C.red} small />
                    <Stat label="Expected gain" value={`+$${Math.round(capped * Math.abs(m.edge) / 100)}`} color={C.green} small />
                  </div>
                  {isCapped && (
                    <div style={{ fontSize: 10, color: C.orange, marginTop: 6, fontFamily: "'JetBrains Mono', monospace" }}>
                      ⚠ Kelly suggests ${betSize} but capped at 5% max (${ Math.round(maxBet)}) to protect bankroll
                    </div>
                  )}
                </div>
              );
            })}

            <div style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, padding: 14, marginTop: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Risk Rules (Hardcoded)</div>
              <div style={{ fontSize: 12, color: C.text2, lineHeight: 1.8 }}>
                <div>● Max single position: <span style={{ color: C.orange, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>5%</span> of bankroll</div>
                <div>● Max total exposure: <span style={{ color: C.yellow, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>20%</span> of bankroll</div>
                <div>● Daily loss limit: <span style={{ color: C.red, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>3%</span> of bankroll → halt all trading</div>
                <div>● Correlation cap: max <span style={{ color: C.blue, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>3</span> positions in same category</div>
                <div>● Time stop: close if no resolution within <span style={{ color: C.purple, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>10 days</span></div>
                <div>● After 3 consecutive losses → <span style={{ color: C.red, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>reduce Kelly to 50%</span> for next 5 trades</div>
              </div>
            </div>
          </div>
        )}

        {/* ─── TRADE HISTORY TAB ─── */}
        {tab === "history" && (
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Trade History & Post-Mortem Analysis</div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 8, marginBottom: 12 }}>
              <Stat label="Total trades" value={TRADE_HISTORY.length} />
              <Stat label="Wins" value={TRADE_HISTORY.filter(t => t.result === "win").length} color={C.green} />
              <Stat label="Losses" value={TRADE_HISTORY.filter(t => t.result === "loss").length} color={C.red} />
              <Stat label="Win rate" value={`${winRate}%`} color={winRate >= 50 ? C.green : C.red} />
              <Stat label="Total P&L" value={`$${totalPnl > 0 ? "+" : ""}${totalPnl}`} color={totalPnl >= 0 ? C.green : C.red} />
              <Stat label="Avg win" value={`+$${Math.round(TRADE_HISTORY.filter(t => t.result === "win").reduce((a, t) => a + t.pnl, 0) / TRADE_HISTORY.filter(t => t.result === "win").length)}`} color={C.green} />
            </div>

            {TRADE_HISTORY.map(t => (
              <div key={t.id} style={{ background: C.surface2, border: `1px solid ${t.result === "win" ? C.green + "33" : C.red + "33"}`, borderRadius: 8, padding: 14, marginBottom: 8, cursor: t.postMortem ? "pointer" : "default" }} onClick={() => t.postMortem && setExpandedTrade(expandedTrade === t.id ? null : t.id)}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, flexWrap: "wrap", gap: 6 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <Badge color={t.result === "win" ? C.green : C.red}>{t.result.toUpperCase()}</Badge>
                    <span style={{ fontSize: 13, fontWeight: 500 }}>{t.market}</span>
                  </div>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 14, color: t.pnl >= 0 ? C.green : C.red }}>{t.pnl > 0 ? "+" : ""}${t.pnl}</span>
                </div>
                <div style={{ display: "flex", gap: 16, fontSize: 11, color: C.text3, fontFamily: "'JetBrains Mono', monospace" }}>
                  <span>Side: {t.side}</span>
                  <span>Entry: {t.entry}¢</span>
                  <span>Exit: {t.exit}¢</span>
                  <span>Size: ${t.size}</span>
                  <span>{t.date}</span>
                </div>
                {t.postMortem && expandedTrade === t.id && (
                  <div style={{ marginTop: 10, padding: 12, background: C.surface3, borderRadius: 6, borderLeft: `3px solid ${C.red}` }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: C.red, marginBottom: 4, fontFamily: "'JetBrains Mono', monospace", letterSpacing: 0.5, textTransform: "uppercase" }}>Post-Mortem Analysis</div>
                    <div style={{ fontSize: 12, color: C.text2, lineHeight: 1.7 }}>{t.postMortem}</div>
                  </div>
                )}
                {t.postMortem && expandedTrade !== t.id && (
                  <div style={{ fontSize: 10, color: C.text3, marginTop: 6, fontFamily: "'JetBrains Mono', monospace" }}>▸ Click to view post-mortem</div>
                )}
              </div>
            ))}

            {/* Aggregate lessons */}
            <div style={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, padding: 14, marginTop: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Aggregate Lessons from Losses</div>
              <div style={{ fontSize: 12, color: C.text2, lineHeight: 1.8 }}>
                <div style={{ marginBottom: 6 }}><span style={{ color: C.red, fontWeight: 600 }}>1.</span> Entered before catalyst confirmation 2/3 times — implement mandatory "catalyst confirmed" checkbox before entry</div>
                <div style={{ marginBottom: 6 }}><span style={{ color: C.red, fontWeight: 600 }}>2.</span> Took 1 trade below confidence threshold — system must hard-block sub-threshold trades</div>
                <div style={{ marginBottom: 6 }}><span style={{ color: C.red, fontWeight: 600 }}>3.</span> Crypto-geo correlation assumption failed — halve crypto position sizes on geopolitical plays</div>
                <div><span style={{ color: C.green, fontWeight: 600 }}>✓</span> Wins came from high-confidence, catalyst-confirmed, single-direction bets (oil up, XOM up). Stick to the edge.</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ padding: "12px 20px", borderTop: `1px solid ${C.border}`, background: C.surface }}>
        <div style={{ fontSize: 10, color: C.text3, lineHeight: 1.6, fontFamily: "'JetBrains Mono', monospace" }}>
          ORACLE v0.2 · Live: Polymarket Gamma + Kalshi /events + Reddit (OAuth+VADER) · XGBoost training not yet wired (next phase: ingest historical resolutions) · Twitter/X paid tier not connected · Not financial advice
        </div>
      </div>
    </div>
  );
}
