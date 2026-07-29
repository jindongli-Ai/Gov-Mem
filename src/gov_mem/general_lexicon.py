"""General-purpose semantic lexicon used only for source-quality hints.

These terms describe ordinary object and value categories.  They are not
dataset entities, answer values, evaluator labels, or authorization rules.
The lexicon helps distinguish an object name from a qualified field value
while the observable state and policy engine remain authoritative.
"""

from __future__ import annotations

import re


GENERAL_OBJECT_LEXICON: dict[str, tuple[str, ...]] = {
    "person": (
        "person", "user", "member", "client", "customer", "patient", "student",
        "teacher", "researcher", "visitor", "contact", "guest", "resident",
    ),
    "identity": (
        "identity", "account holder", "profile", "principal", "owner", "role",
        "name", "badge", "credential", "identifier",
    ),
    "relationship": (
        "family", "parent", "child", "partner", "spouse", "relative", "friend",
        "colleague", "coworker", "classmate", "advisor", "supervisor", "delegate",
    ),
    "organization": (
        "organization", "company", "school", "university", "hospital", "clinic",
        "government", "agency", "laboratory", "lab", "department", "team", "group",
    ),
    "work": (
        "job", "employment", "employer", "employee", "workplace", "office",
        "shift", "meeting", "manager", "colleague", "assignment", "position",
        "career", "profession", "occupation", "payroll", "leave",
    ),
    "education": (
        "course", "class", "lesson", "lecture", "cohort", "section", "enrollment",
        "application", "curriculum", "pathway", "school", "campus", "teacher",
        "student", "textbook", "tuition", "semester", "term",
    ),
    "academic": (
        "major", "minor", "degree", "semester", "thesis", "dissertation",
        "advisor", "faculty", "exam", "grade", "assignment", "scholarship",
        "credit", "transcript", "graduation", "admission",
    ),
    "research": (
        "research", "experiment", "hypothesis", "protocol", "sample", "dataset",
        "paper", "manuscript", "publication", "study", "trial", "laboratory",
        "cohort", "finding", "result", "researcher", "review", "replication",
    ),
    "medical": (
        "appointment", "visit", "trial", "care", "service", "referral", "claim",
        "treatment", "clinic", "patient", "provider", "chart", "prescription",
        "symptom", "diagnosis", "laboratory", "imaging", "therapy", "procedure",
    ),
    "health": (
        "health", "wellness", "symptom", "condition", "disease", "recovery",
        "exercise", "sleep", "diet", "nutrition", "allergy", "vaccine",
    ),
    "management": (
        "project", "program", "initiative", "campaign", "workstream", "task",
        "milestone", "deliverable", "workspace", "team", "group", "department",
        "organization", "company", "case", "matter", "ticket", "incident",
    ),
    "household": (
        "household", "home", "residence", "family", "pet", "vehicle", "car",
        "parcel", "package", "delivery", "order", "reservation", "booking",
        "trip", "event", "membership", "subscription", "property", "lease",
    ),
    "housing": (
        "home", "house", "apartment", "residence", "room", "kitchen", "bedroom",
        "address", "lease", "rent", "landlord", "tenant", "maintenance",
    ),
    "transport": (
        "ride", "pickup", "drop-off", "vehicle", "car", "bus", "train", "station",
        "route", "ticket", "flight", "airport", "parking", "driver",
    ),
    "travel": (
        "trip", "travel", "hotel", "reservation", "booking", "itinerary", "visa",
        "destination", "luggage", "check-in", "departure", "arrival",
    ),
    "shopping": (
        "product", "item", "order", "store", "shop", "purchase", "cart", "return",
        "refund", "warranty", "receipt",
    ),
    "service": (
        "service", "request", "support", "customer", "provider", "repair", "claim",
        "application", "subscription", "membership",
    ),
    "records": (
        "record", "file", "folder", "thread", "message", "request", "form",
        "profile", "conversation", "document", "note", "entry", "item", "resource",
    ),
    "finance": (
        "account", "budget", "invoice", "payment", "expense", "claim", "balance",
        "price", "cost", "fee", "salary", "wage", "discount", "rate", "tax",
    ),
    "economics": (
        "income", "revenue", "profit", "loss", "debt", "loan", "savings", "market",
        "demand", "supply", "funding", "grant", "capital", "asset", "liability",
    ),
    "payment": (
        "payment", "pay", "transfer", "refund", "invoice", "bill", "receipt",
        "deposit", "withdrawal", "transaction", "checkout",
    ),
    "contract": (
        "contract", "agreement", "vendor", "client", "customer", "renewal", "term",
        "clause", "signature", "obligation", "license", "permit",
    ),
    "technology": (
        "device", "phone", "computer", "software", "system", "server", "network",
        "database", "model", "app", "application", "website", "login",
    ),
    "document": (
        "document", "report", "letter", "form", "certificate", "transcript", "paper",
        "note", "record", "file", "folder", "spreadsheet",
    ),
    "food": (
        "food", "meal", "restaurant", "menu", "recipe", "ingredient", "allergy",
        "diet", "nutrition", "drink", "grocery",
    ),
    "sport": (
        "sport", "game", "match", "team", "training", "coach", "player", "league",
        "stadium", "fitness",
    ),
    "entertainment": (
        "movie", "film", "music", "concert", "show", "book", "game", "festival",
        "ticket", "performance",
    ),
    "environment": (
        "weather", "air", "water", "energy", "waste", "pollution", "temperature",
        "climate", "noise", "recycling",
    ),
    "governance": (
        "policy", "rule", "permission", "delegation", "approval", "authorization",
        "audit", "review", "compliance", "restriction", "exception", "appeal",
    ),
}


# Broad semantic topics used by both query parsing and observable-state
# construction.  These names are deliberately domain-generic: they describe
# ordinary language categories, never GateMem cases, expected actions, or
# evaluator labels.
GENERAL_TOPIC_LEXICON: dict[str, tuple[str, ...]] = {
    "logistics": (
        "ride", "rides", "pickup", "pick up", "drop-off", "drop off",
        "transportation", "arrival", "delivery", "travel", "route", "shipment",
        "check-in", "check in", "handoff", "dropoff", "dispatch", "courier",
        "shipment", "parcel", "package", "stocking", "overflow", "handoff point",
    ),
    "scheduling": (
        "appointment", "calendar", "schedule", "visit", "meeting", "booking",
        "follow-up", "follow up", "time", "deadline", "date", "when", "slot",
    ),
    "communication": (
        "callback", "call back", "return-call", "return call", "phone", "mobile",
        "contact", "voicemail", "message", "email", "portal", "notification",
    ),
    "medication": (
        "medicine", "medication", "prescription", "drug", "dose", "dosage", "tablet",
        "pill", "refill", "therapy", "treatment",
    ),
    "health": (
        "diagnosis", "clinical", "symptom", "condition", "treatment", "procedure",
        "health", "care", "disease", "recovery", "wellness", "nutrition",
    ),
    "medical": (
        "medical", "medicine", "medication", "patient", "provider", "clinic", "hospital",
        "appointment", "treatment", "therapy", "prescription", "diagnosis",
    ),
    "laboratory": (
        "lab", "laboratory", "test result", "bloodwork", "screening", "measurement",
        "sample", "assay", "result",
    ),
    "imaging": (
        "imaging", "scan", "ultrasound", "mri", "biopsy", "pathology", "x-ray",
        "radiology",
    ),
    "safety": (
        "red flag", "trigger", "warning", "risk", "urgent", "emergency", "danger",
        "incident", "hazard", "alert",
    ),
    "access_control": (
        "access", "permission", "authorization", "authorize", "revoke", "share",
        "private", "confidential", "credential", "credentials", "code", "password",
        "pin", "token", "secret", "restricted",
    ),
    "identity": (
        "identity", "name", "account", "profile", "same person", "same name",
        "who", "identifier",
    ),
    "location": (
        "address", "place", "location", "room", "site", "where", "building", "floor",
        "desk", "hall", "suite", "station", "venue", "cubby", "shelf", "shelves",
    ),
    "finance": (
        "budget", "payment", "invoice", "billing", "reimbursement", "cost", "price",
        "amount", "discount", "salary", "wage", "expense", "financial", "bank", "tax",
    ),
    "economics": (
        "income", "revenue", "profit", "loss", "debt", "loan", "savings", "funding",
        "capital", "asset", "liability", "market",
    ),
    "legal": (
        "contract", "nda", "non-disclosure", "legal", "counsel", "litigation",
        "settlement", "compliance", "agreement", "license", "permit",
    ),
    "privacy": (
        "private", "confidential", "personal", "sensitive", "internal", "restricted",
        "not for sharing", "do not disclose", "privacy",
    ),
    "work": (
        "job", "employment", "employer", "employee", "workplace", "shift", "manager",
        "colleague", "career", "profession", "payroll", "leave",
    ),
    "education": (
        "school", "course", "class", "lesson", "lecture", "teacher", "student",
        "campus", "enrollment", "curriculum", "tuition",
    ),
    "academic": (
        "major", "minor", "degree", "semester", "thesis", "dissertation", "advisor",
        "faculty", "exam", "grade", "credit", "transcript", "graduation",
    ),
    "research": (
        "research", "experiment", "hypothesis", "protocol", "sample", "dataset", "paper",
        "manuscript", "publication", "study", "finding", "replication",
    ),
    "household": (
        "household", "home", "family", "pet", "vehicle", "parcel", "package", "order",
        "reservation", "membership", "subscription", "property", "lease",
    ),
    "technology": (
        "device", "phone", "computer", "software", "system", "server", "network",
        "database", "model", "application", "website", "login",
    ),
    "document": (
        "document", "report", "letter", "form", "certificate", "transcript", "paper",
        "note", "record", "file", "folder", "spreadsheet",
    ),
    "food": (
        "food", "meal", "restaurant", "menu", "recipe", "ingredient", "allergy",
        "diet", "nutrition", "drink", "grocery",
    ),
    "sport": (
        "sport", "game", "match", "team", "training", "coach", "player", "league",
        "stadium", "fitness",
    ),
    "entertainment": (
        "movie", "film", "music", "concert", "show", "book", "game", "festival",
        "ticket", "performance",
    ),
    "environment": (
        "weather", "air", "water", "energy", "waste", "pollution", "temperature",
        "climate", "noise", "recycling",
    ),
}


def lexicon_terms_match(text: str, term: str) -> bool:
    """Match a lexical item as a word/phrase, avoiding substring collisions."""
    value = str(text or "").casefold()
    pattern = re.escape(str(term).casefold()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", value) is not None


def topics_from_text(text: str) -> tuple[str, ...]:
    topics = {
        topic for topic, terms in GENERAL_TOPIC_LEXICON.items()
        if any(lexicon_terms_match(text, term) for term in terms)
    }
    # A time range is an ordinary scheduling signal even when the sentence
    # does not contain the word "schedule" or "window".
    if re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b\s*(?:to|[-–—])\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b", str(text or ""), re.IGNORECASE):
        topics.add("scheduling")
    return tuple(sorted(topics))


GENERAL_VALUE_HEAD_LEXICON: dict[str, tuple[str, ...]] = {
    "time": (
        "date", "day", "time", "window", "deadline", "target", "arrival",
        "departure", "duration", "frequency", "period", "calendar", "schedule",
        "appointment", "milestone",
    ),
    "location": (
        "bay", "room", "site", "address", "desk", "hall", "suite", "floor",
        "building", "office", "station", "entrance", "door", "route", "stop",
        "parking", "location", "place", "venue", "destination",
    ),
    "access": (
        "badge", "code", "credential", "password", "passcode", "pin", "token",
        "key", "login", "access", "scope", "permission", "authorization", "approval",
    ),
    "identity": (
        "identity", "account", "role", "owner", "name", "profile", "principal",
    ),
    "finance": (
        "amount", "budget", "price", "cost", "fee", "payment", "invoice",
        "discount", "rate", "salary", "wage", "expense", "cap", "limit",
        "vendor", "term", "renewal", "deposit", "balance", "total",
    ),
    "economics": (
        "income", "revenue", "profit", "loss", "debt", "loan", "savings", "funding",
        "capital", "asset", "liability", "market", "demand", "supply",
    ),
    "state": (
        "status", "state", "label", "color", "stage", "phase", "blocker",
        "issue", "risk", "condition", "scope", "change", "update", "reason", "summary",
    ),
    "management": (
        "instruction", "recommendation", "action", "plan", "task", "milestone",
    ),
    "health": (
        "diagnosis", "condition", "symptom", "treatment", "medication", "dose",
        "test", "result", "measurement", "imaging", "support", "care", "referral",
    ),
}


GENERAL_OBJECT_PREFIXES = frozenset(
    term for terms in GENERAL_OBJECT_LEXICON.values() for term in terms
)
GENERAL_VALUE_HEADS = frozenset(
    term for terms in GENERAL_VALUE_HEAD_LEXICON.values() for term in terms
)
