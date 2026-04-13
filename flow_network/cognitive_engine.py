import torch
import torch.nn.functional as F
import collections
import re
import time
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field


# ============================================================================
# CONCEPT — Podstawowa jednostka znaczenia (z ConceptualHCM)
# ============================================================================

@dataclass
class Concept:
    """
    Koncept = „znak większej całości"
    Ma esencję, powiązania, wagę i możliwość rozpakowania.
    """
    id: str
    essence: str
    keywords: Set[str] = field(default_factory=set)
    links: Set[str] = field(default_factory=set)
    weight: float = 1.0
    frequency: int = 1
    turns: List[int] = field(default_factory=list)
    raw_content: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def activate(self, strength: float = 1.0):
        """Aktywacja konceptu — wzmocnienie wagi"""
        self.weight = min(1.0, self.weight + 0.1 * strength)

    def decay(self, rate: float = 0.95):
        """Temporal decay — stare koncepty tracą wagę"""
        self.weight *= rate

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'essence': self.essence,
            'keywords': list(self.keywords),
            'links': list(self.links),
            'weight': self.weight,
            'frequency': self.frequency
        }


# ============================================================================
# CONCEPT GRAPH — Sieć powiązań z spreading activation
# ============================================================================

class ConceptGraph:
    """
    Graf konceptów ze spreading activation, temporal decay
    i wyszukiwaniem semantycznym. Zastępuje prymitywny KnowledgeGraph.
    """

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.edges: Dict[Tuple[str, str], float] = {}

    def add_concept(self, concept: Concept):
        """Dodaj koncept do grafu (merguj jeśli istnieje)"""
        if concept.id in self.concepts:
            existing = self.concepts[concept.id]
            existing.frequency += concept.frequency
            existing.weight = max(existing.weight, concept.weight)
            existing.keywords.update(concept.keywords)
            existing.turns.extend(concept.turns)
            existing.raw_content.extend(concept.raw_content[-3:])  # limit surowego tekstu
        else:
            self.concepts[concept.id] = concept

    def add_fact(self, subject: str, predicate: str, object_node: str):
        """Dodaj fakt jako koncepty + krawędź (kompatybilność z KnowledgeGraph API)"""
        subj_id = f"ENTITY_{subject.upper().replace(' ', '_')}"
        obj_id = f"ENTITY_{object_node.upper().replace(' ', '_')}"

        self.add_concept(Concept(
            id=subj_id, essence=subject,
            keywords={subject.lower()}, weight=0.8
        ))
        self.add_concept(Concept(
            id=obj_id, essence=object_node,
            keywords={object_node.lower()}, weight=0.7
        ))
        self.add_edge(subj_id, obj_id, 0.8)

    def add_edge(self, id1: str, id2: str, strength: float = 1.0):
        """Dodaj połączenie między konceptami"""
        if id1 in self.concepts and id2 in self.concepts:
            edge = tuple(sorted([id1, id2]))
            self.edges[edge] = self.edges.get(edge, 0.0) + strength
            self.concepts[id1].links.add(id2)
            self.concepts[id2].links.add(id1)

    def get_neighbors(self, concept_id: str, min_strength: float = 0.3) -> List[str]:
        """Znajdź sąsiadów konceptu"""
        neighbors = []
        for (id1, id2), strength in self.edges.items():
            if strength >= min_strength:
                if id1 == concept_id:
                    neighbors.append(id2)
                elif id2 == concept_id:
                    neighbors.append(id1)
        return neighbors

    def calculate_dissonance(self, id1: str, id2: str) -> float:
        """
        Oblicz Napięcie Poznawcze (Dissonance) między dwoma konceptami bazując na
        indeksie Jaccarda ich sąsiadów.
        0.0 -> pełne nakładanie się sąsiadów (nuda/tautologia)
        1.0 -> brak wspólnych sąsiadów (potencjalna halucynacja)
        0.4-0.7 -> strefa kreatywności (synteza)
        """
        if id1 not in self.concepts or id2 not in self.concepts:
            return 1.0  # Maksymalny dysonans jeśli brakuje w grafie

        neighbors_1 = set(self.get_neighbors(id1))
        neighbors_2 = set(self.get_neighbors(id2))
        
        # Jeśli którykolwiek koncept nie ma sąsiadów (poza sobą), uznajemy, że wiedza jest mętna
        if not neighbors_1 and not neighbors_2:
            return 1.0
            
        intersection = len(neighbors_1.intersection(neighbors_2))
        union = len(neighbors_1.union(neighbors_2))
        
        if union == 0:
            return 1.0
            
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

    def activate_subgraph(self, seed_ids: List[str], depth: int = 2) -> Set[str]:
        """
        Spreading activation — aktywuj podgraf wokół seed konceptów.
        Jak myśli rozprzestrzeniają się w umyśle.
        """
        activated = set(seed_ids)
        frontier = set(seed_ids)

        for _ in range(depth):
            new_frontier = set()
            for concept_id in frontier:
                if concept_id in self.concepts:
                    self.concepts[concept_id].activate(0.5)
                neighbors = self.get_neighbors(concept_id)
                new_frontier.update(neighbors)
            activated.update(new_frontier)
            frontier = new_frontier

        return activated

    def retrieve_context(self, query_entities: List[str]) -> str:
        """Pobierz kontekst (kompatybilność z KnowledgeGraph API + spreading activation)"""
        # Znajdź matching koncepty
        seed_ids = []
        for entity in query_entities:
            entity_lower = entity.lower()
            for cid, concept in self.concepts.items():
                if entity_lower in concept.keywords or entity_lower in concept.essence.lower():
                    seed_ids.append(cid)

        if not seed_ids:
            return ""

        # Spreading activation
        activated_ids = self.activate_subgraph(seed_ids, depth=1)

        # Zbuduj kontekst z aktywowanych konceptów
        context_parts = []
        for cid in activated_ids:
            if cid in self.concepts:
                c = self.concepts[cid]
                links_str = ", ".join(list(c.links)[:3])
                context_parts.append(f"{c.essence} [w={c.weight:.2f}, links: {links_str}]")

        return " | ".join(context_parts) if context_parts else ""

    def apply_temporal_decay(self, current_turn: int, rate: float = 0.98):
        """Zastosuj temporal decay do konceptów nieaktywnych w bieżącej turze"""
        for concept in self.concepts.values():
            if current_turn not in concept.turns:
                concept.decay(rate)

    def get_top_concepts(self, n: int = 10) -> List[Concept]:
        """Najważniejsze koncepty według wagi × częstotliwości"""
        sorted_concepts = sorted(
            self.concepts.values(),
            key=lambda c: c.weight * c.frequency,
            reverse=True
        )
        return sorted_concepts[:n]

    def get_summary(self) -> str:
        """Zwróć mapę konceptów jako tekst"""
        top = self.get_top_concepts(8)
        parts = ["[CONCEPT_MAP]"]
        for c in top:
            parts.append(f"  {c.id}: {c.essence} (w={c.weight:.2f}, freq={c.frequency})")
        return "\n".join(parts)


# ============================================================================
# CONCEPT EXTRACTOR — Ekstrakcja konceptów z tekstu
# ============================================================================

class ConceptExtractor:
    """Wyciąga koncepty z tekstu — NER-like + keyword extraction"""

    PATTERNS = {
        'PERSON': r'(?i)(?:jestem|nazywam się|mam na imię|i\'m|my name is)\s+([A-Z][a-z]+)',
        'PROJECT': r'(?i)(projekt[a-z]*|project)\s+([A-Za-z_]+)',
        'TECH': r'\b(Python|JavaScript|React|TensorFlow|PyTorch|AI|ML|API|CUDA|GPU|LLM|Flow)\b',
        'ACTION': r'(?i)\b(tworzę|buduję|pracuję nad|rozwijam|uczę|build|create|train)\b',
    }

    STOPWORDS = {
        'i', 'a', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'może', 'bardzo', 'jest', 'był', 'będzie', 'to', 'w', 'na',
        'z', 'o', 'że', 'jak', 'co', 'dla', 'nie', 'się', 'ale'
    }

    @classmethod
    def extract_keywords(cls, text: str, top_n: int = 5) -> Set[str]:
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in cls.STOPWORDS and len(w) > 3]
        counter = Counter(words)
        return set(w for w, _ in counter.most_common(top_n))

    @classmethod
    def extract_concepts_from_text(cls, text: str, turn_id: int = 0) -> List[Concept]:
        """Wyciągnij koncepty z tekstu"""
        concepts = []

        for entity_type, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    value = match[-1] if isinstance(match, tuple) else match
                    if len(value) < 2:
                        continue
                    concept_id = f"{entity_type}_{value.upper().replace(' ', '_')}"
                    weight = 0.9 if entity_type in ['PERSON', 'PROJECT'] else 0.7
                    concepts.append(Concept(
                        id=concept_id, essence=f"{entity_type}: {value}",
                        keywords={value.lower()}, turns=[turn_id],
                        raw_content=[text[:100]], weight=weight
                    ))

        # Fallback: keywords jako koncepty
        if len(concepts) < 2:
            keywords = cls.extract_keywords(text, top_n=2)
            for kw in keywords:
                concepts.append(Concept(
                    id=f"TOPIC_{kw.upper()}", essence=f"Topic: {kw}",
                    keywords={kw}, turns=[turn_id],
                    raw_content=[text[:100]], weight=0.4
                ))

        return concepts


# ============================================================================
# SELF-CRITIQUE MODULE — Samokrytycyzm odpowiedzi (Tier 2)
# ============================================================================

class SelfCritiqueModule:
    """
    Moduł samokrytycyzmu. Ocenia jakość wygenerowanych odpowiedzi
    na podstawie koherencji, powtórzeń i spójności z kontekstem.
    Bayesowska aktualizacja wag wzorców.
    """

    def __init__(self, decay_rate: float = 0.95):
        self.critique_history: List[Dict] = []
        self.decay_rate = decay_rate
        self.quality_mean = 0.5  # Prior: średnia jakość
        self.quality_var = 0.25  # Prior: wariancja

    def critique(self, generated_text: str, context: str = "") -> Dict:
        """Oceniaj wygenerowany tekst"""
        scores = {}

        # 1. Repetition penalty — sprawdź powtórzenia n-gramów
        words = generated_text.split()
        if len(words) > 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            unique_ratio = len(set(trigrams)) / max(len(trigrams), 1)
            scores['repetition'] = unique_ratio
        else:
            scores['repetition'] = 1.0

        # 2. Length quality — zbyt krótkie lub zbyt długie odpowiedzi
        length = len(generated_text.strip())
        if length < 5:
            scores['length'] = 0.1
        elif length > 2000:
            scores['length'] = 0.5
        else:
            scores['length'] = min(1.0, length / 50)

        # 3. Context relevance — ile słów z kontekstu pojawia się w odpowiedzi
        if context:
            context_words = set(re.findall(r'\b\w{4,}\b', context.lower()))
            response_words = set(re.findall(r'\b\w{4,}\b', generated_text.lower()))
            if context_words:
                overlap = len(context_words & response_words) / len(context_words)
                scores['relevance'] = min(1.0, overlap * 3)
            else:
                scores['relevance'] = 0.5
        else:
            scores['relevance'] = 0.5

        # 4. Coherence — sprawdź czy tekst nie jest chaotyczny
        if len(generated_text) > 10:
            char_diversity = len(set(generated_text)) / len(generated_text)
            scores['coherence'] = min(1.0, char_diversity * 3)
        else:
            scores['coherence'] = 0.5

        # 5. Cognitive Tension (Dissonance Philosophy z ARCHEUS)
        # Obliczamy abstrakcyjny wektor dysonansu bazując na stosunku nieznanych słów do znanych
        if context:
            all_generated = set(re.findall(r'\b\w{4,}\b', generated_text.lower()))
            if all_generated:
                dissonance = 1.0 - (len(context_words & all_generated) / len(all_generated))
                
                # Nagradzaj strefę "Kreatywności/Syntezy" (0.4 - 0.7) a karaj za nudę (0.0) i halucynacje (1.0)
                if 0.4 <= dissonance <= 0.7:
                    scores['cognitive_tension'] = 1.0  # Idealny stan syntezy
                else:
                    # Im dalej od złotego środka (0.55), tym niższy wynik
                    scores['cognitive_tension'] = max(0.0, 1.0 - 2.0 * abs(dissonance - 0.55))
            else:
                scores['cognitive_tension'] = 0.5
        else:
            scores['cognitive_tension'] = 0.5

        # Overall quality score
        overall = sum(scores.values()) / len(scores)
        scores['overall'] = overall

        # Bayesian update
        self._update_quality_prior(overall)

        # Store critique
        self.critique_history.append({
            'timestamp': time.time(),
            'scores': scores,
            'text_length': len(generated_text)
        })

        return scores

    def _update_quality_prior(self, observation: float, obs_var: float = 0.1):
        """Bayesowska aktualizacja jakości"""
        prior_prec = 1.0 / max(1e-9, self.quality_var)
        like_prec = 1.0 / max(1e-9, obs_var)
        post_var = 1.0 / (prior_prec + like_prec)
        post_mean = post_var * (self.quality_mean * prior_prec + observation * like_prec)
        self.quality_mean = post_mean
        self.quality_var = post_var

    def should_regenerate(self, scores: Dict, threshold: float = 0.3) -> bool:
        """Czy odpowiedź jest na tyle słaba, że warto spróbować ponownie?"""
        return scores.get('overall', 1.0) < threshold

    def get_quality_trend(self) -> str:
        """Zwróć trend jakości"""
        if len(self.critique_history) < 2:
            return "insufficient_data"
        recent = [h['scores']['overall'] for h in self.critique_history[-10:]]
        if len(recent) >= 3:
            trend = recent[-1] - recent[0]
            if trend > 0.1:
                return "improving"
            elif trend < -0.1:
                return "declining"
        return "stable"


# ============================================================================
# EPISODIC BUFFER — Pamięć Epizodyczna
# ============================================================================

class EpisodicBuffer:
    """Pamięć Epizodyczna z Ograniczoną Pojemnością + metadata"""

    def __init__(self, capacity: int = 10):
        self.events = collections.deque(maxlen=capacity)

    def add_event(self, event: str, metadata: Optional[Dict] = None):
        self.events.append({
            'text': event,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })

    def get_recent_history(self, n: Optional[int] = None) -> str:
        items = list(self.events)
        if n is not None:
            items = items[-n:]
        return " | ".join(e['text'] for e in items)


# ============================================================================
# COGNITIVE FLOW AGENT — Agent Kognitywny (rozszerzony)
# ============================================================================

class CognitiveFlowAgent:
    """
    Agent Kognitywny spinający FlowNetwork (Pamięć Robocza)
    z ConceptGraph (Pamięć Semantyczna), EpisodicBuffer (Epizodyczna)
    i SelfCritiqueModule (Samokrytycyzm).

    Usprawnienia względem oryginału:
    - ConceptGraph ze spreading activation zamiast prostego KnowledgeGraph
    - Concept Extractor — automatyczna ekstrakcja konceptów z tekstu
    - Self-Critique — ocena własnych odpowiedzi z Bayesowskim feedbackiem
    - Temporal decay — stare koncepty tracą wagę
    """

    def __init__(self, flow_model, stoi: Dict[str, int], itos: Dict[int, str], device='cpu'):
        self.brain = flow_model
        self.stoi = stoi
        self.itos = itos
        self.device = device

        # Pamięć Semantyczna — ConceptGraph ze spreading activation
        self.semantic_memory = ConceptGraph()
        # Pamięć Epizodyczna — z metadanymi
        self.episodic_memory = EpisodicBuffer(capacity=10)
        # Moduł samokrytycyzmu
        self.self_critique = SelfCritiqueModule()

        self._turn_counter = 0

    def _encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, self.stoi.get(' ', 0)) for c in text]

    def _decode(self, tokens: List[int]) -> str:
        return ''.join([self.itos.get(i, '?') for i in tokens])

    def perceive_and_think(self, user_input: str, extracted_keywords: List[str] = None,
                           max_retries: int = 2, trusted_source: bool = False) -> str:
        """
        Pętla Kognitywna:
        1. Ekstrakcja konceptów z wejścia użytkownika
        2. Spreading activation w grafie wiedzy
        3. Scalenie Pamięci Epizodycznej + Semantycznej z Roboczą
        4. Generacja predykcji
        5. Self-Critique — ocena i ewentualna regeneracja (pomijane jeśli trusted_source=True)
        """
        self._turn_counter += 1

        # 1. Automatyczna ekstrakcja konceptów z wejścia
        new_concepts = ConceptExtractor.extract_concepts_from_text(
            user_input, turn_id=self._turn_counter
        )
        for concept in new_concepts:
            self.semantic_memory.add_concept(concept)

        # Powiąż koncepty z tego samego turnu
        if len(new_concepts) > 1:
            for i, c1 in enumerate(new_concepts):
                for c2 in new_concepts[i+1:]:
                    self.semantic_memory.add_edge(c1.id, c2.id, 0.6)

        # 2. Temporal decay
        self.semantic_memory.apply_temporal_decay(self._turn_counter)

        # 3. Zapisz epizod
        self.episodic_memory.add_event(
            f"User asked: {user_input[:40]}...",
            metadata={'turn': self._turn_counter, 'concepts': [c.id for c in new_concepts]}
        )

        # 4. Pobierz kontekst z RAG (spreading activation)
        if extracted_keywords is None:
            extracted_keywords = []
            for word in user_input.split():
                if word and (word[0].isupper() or len(word) > 5):
                    extracted_keywords.append(word)

        hard_facts = self.semantic_memory.retrieve_context(extracted_keywords)

        # 5. Złóż pełny kontekst roboczy
        system_prompt = ""
        if hard_facts:
            system_prompt += f"[SEMANTIC_RAG: {hard_facts}] "

        recent_history = self.episodic_memory.get_recent_history(n=5)
        system_prompt += f"[EPISODIC: {recent_history}] "

        full_working_memory = system_prompt + user_input

        # 6. Generacja
        generated = self._generate_from_brain(full_working_memory, max_tokens=100)

        # 7. Ekstremalne przyspieszenie: Turbo-RAG (Trusted Source)
        if trusted_source:
            self.episodic_memory.add_event(
                f"[TRUSTED] Generated: {generated[:30]}...",
                metadata={'quality': 1.0, 'retries': 0, 'trusted': True}
            )
            return generated

        # 8. Self-Critique (jeśli nie jest to trusted_source)
        critique_scores = self.self_critique.critique(generated, context=user_input)

        # 9. Retry jeśli jakość zbyt niska
        retries = 0
        while self.self_critique.should_regenerate(critique_scores) and retries < max_retries:
            retries += 1
            generated = self._generate_from_brain(full_working_memory, max_tokens=100)
            critique_scores = self.self_critique.critique(generated, context=user_input)

        # 10. Zapamiętaj
        self.episodic_memory.add_event(
            f"Generated (quality={critique_scores['overall']:.2f}, retries={retries}): {generated[:30]}...",
            metadata={'quality': critique_scores['overall'], 'retries': retries}
        )

        return generated

    @torch.no_grad()
    def _generate_from_brain(self, full_context: str, max_tokens: int) -> str:
        self.brain.eval()
        encoded = self._encode(full_context)
        idx = torch.tensor(encoded, dtype=torch.long, device=self.device).unsqueeze(0)

        for _ in range(max_tokens):
            idx_cond = idx[:, -256:]
            logits, _ = self.brain(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        out = self._decode(idx[0].tolist())
        self.brain.train()

        generated_part = out[len(full_context):]
        return generated_part

    def dream(self):
        """
        Pętla Snu (Consolidation Layer).
        Konsoliduje pamięć epizodyczną do grafu konceptów.
        """
        history = self.episodic_memory.get_recent_history()

        # Wyciągnij koncepty z historii epizodycznej
        dream_concepts = ConceptExtractor.extract_concepts_from_text(
            history, turn_id=self._turn_counter
        )
        for concept in dream_concepts:
            concept.weight *= 1.2  # Wzmocnienie konsolidowanych konceptów
            self.semantic_memory.add_concept(concept)

        concept_summary = self.semantic_memory.get_summary()
        quality_trend = self.self_critique.get_quality_trend()

        return (
            f"[DREAM CONSOLIDATION]\n"
            f"Przeanalizowano {len(dream_concepts)} konceptów z pamięci epizodycznej.\n"
            f"Trend jakości generacji: {quality_trend}\n"
            f"Średnia jakość (Bayes): {self.self_critique.quality_mean:.3f}\n"
            f"{concept_summary}"
        )

    def get_agent_status(self) -> Dict:
        """Zwróć status agenta"""
        return {
            'turn_count': self._turn_counter,
            'concepts_count': len(self.semantic_memory.concepts),
            'edges_count': len(self.semantic_memory.edges),
            'episodic_events': len(self.episodic_memory.events),
            'quality_mean': self.self_critique.quality_mean,
            'quality_trend': self.self_critique.get_quality_trend(),
            'top_concepts': [c.to_dict() for c in self.semantic_memory.get_top_concepts(5)]
        }
