import logging
from datetime import datetime, date
from typing import Optional

from .evidence_collector import EvidenceCollector
from .hypothesis_tracker import HypothesisTracker
from .confidence_engine import ConfidenceEngine
from .contradiction_detector import ContradictionDetector
from .recommendation_engine import RecommendationEngine
from .research_memory import ResearchMemory
from .deployment_advisor import DeploymentAdvisor
from .director_classifier import DirectorClassifier

logger = logging.getLogger("proxima_ops.ard.pipeline")


class DirectorPipeline:
    def __init__(self):
        self.collector = EvidenceCollector()
        self.hypotheses = HypothesisTracker()
        self.confidence = ConfidenceEngine(self.hypotheses)
        self.contradictions = ContradictionDetector()
        self.recommender = RecommendationEngine()
        self.memory = ResearchMemory()
        self.advisor = DeploymentAdvisor(self.confidence)
        self.classifier = DirectorClassifier()

    def daily_report(self) -> dict:
        ev = self.collector.summary()
        hyp = self.hypotheses.all_confidences()
        risk = self.advisor.biggest_risk(hyp)
        strength = self.advisor.biggest_strength(hyp)
        n_trades = self._latest_trade_count()

        if n_trades < 10:
            es = 0.0
            rc = "UNCHANGED"
            dc = "UNCHANGED"
            ate = "COLLECTING_DATA"
            health = "COLLECTING_DATA"
            freq_match = "COLLECTING_DATA"
            ccount = 0
            rec_str = "OBSERVE_LONGER"
            reasons = ["Awaiting sufficient evidence (n_trades < 10)"]
            cls_str = "RESEARCH_PENDING"
        else:
            es = self.confidence.evidence_strength(n_trades, self.memory.count_dailies())
            rc = self.confidence.research_confidence()
            dc = self.deployment_confidence()

            ate = self._latest_ate()
            health = self._latest_health()
            freq_match = self._latest_freq_match()
            ccount = self.contradictions.count()

            rec = self.recommender.evaluate(n_trades, 0.0, ccount, es, rc, dc, ate, health, freq_match)
            rec_advised = self.advisor.recommend(rec, risk, strength)
            cls = self.classifier.classify(n_trades, es, rc, dc, ate, health, ccount)
            rec_str = rec_advised["recommendation"]
            reasons = rec.get("reasons", [])
            cls_str = cls["classification"]

        report = {
            "date": date.today().isoformat(),
            "evidence_strength": es,
            "research_confidence": rc,
            "deployment_confidence": dc,
            "alpha_transfer": ate,
            "health_index": health,
            "freq_match": freq_match,
            "contradictions": ccount,
            "biggest_risk": risk,
            "biggest_strength": strength,
            "recommendation": rec_str,
            "recommendation_reasons": reasons,
            "classification": cls_str,
            "hypotheses": hyp}
        self.memory.store_daily(report)
        return report

    def weekly_report(self) -> dict:
        dailies = self.memory.recent_dailies(7)
        first = dailies[0] if dailies else {}
        last = dailies[-1] if dailies else {}

        gained = {}
        lost = {}
        if first and last and "hypotheses" in first and "hypotheses" in last:
            for k in first["hypotheses"]:
                diff = last["hypotheses"].get(k, 0) - first["hypotheses"].get(k, 0)
                if diff > 0.02:
                    gained[k] = round(diff, 3)
                elif diff < -0.02:
                    lost[k] = round(diff, 3)

        rec_trend = [d.get("recommendation", "NO_ACTION") for d in dailies]
        converging = last.get("alpha_transfer", 0) >= first.get("alpha_transfer", 0) if first and last else False

        report = {
            "date": date.today().isoformat(),
            "week_start": dailies[0]["date"] if dailies else "N/A",
            "gained_confidence": gained,
            "lost_confidence": lost,
            "new_contradictions": self.contradictions.count(),
            "converging": converging,
            "recommendation_trend": rec_trend,
            "latest_recommendation": last.get("recommendation", "NO_ACTION") if last else "NO_ACTION",
            "latest_classification": last.get("classification", "RESEARCH_PENDING") if last else "RESEARCH_PENDING",
            "total_dailies": self.memory.count_dailies()}
        self.memory.store_weekly(report)
        return report

    def deployment_confidence(self) -> float:
        return 0.5

    def _latest_trade_count(self) -> int:
        for e in reversed(self.collector.all_evidence()):
            if "n_trades" in e:
                return e["n_trades"]
        return 0

    def _latest_ate(self) -> float:
        for e in reversed(self.collector.all_evidence()):
            if e.get("source") == "rce" and "ate" in e:
                return e["ate"]
        return 0.0

    def _latest_health(self) -> float:
        for e in reversed(self.collector.all_evidence()):
            if e.get("source") == "rce" and "health_index" in e:
                return e["health_index"]
        return 0.0

    def _latest_freq_match(self) -> float:
        for e in reversed(self.collector.all_evidence()):
            if "ate" in e:
                return max(0.0, min(1.0, e["ate"]))
        return 0.0
