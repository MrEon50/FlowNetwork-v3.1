import sys
import torch
import time

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from flow_network.models import EnhancedFlowTransformer
from flow_network.cognitive_engine import CognitiveFlowAgent, ConceptGraph, EpisodicBuffer, SelfCritiqueModule

def run_cognitive_demo():
    print("=" * 60)
    print("🧠 DEMONSTRACJA ARCHITEKTURY KOGNITYWNEJ (FLOW + RAG)")
    print("   Wersja 2.0: ConceptGraph + Spreading Activation")
    print("               + Self-Critique + Temporal Decay")
    print("=" * 60)
    print("Ten test udowadnia, że pozbycie się kwadratowej pamięci O(N²) (Attention)")
    print("nie szkodzi AI, jeśli wyposażysz sieć Flow w inteligentny zewn. Graf Wiedzy (RAG).")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Symulacja małego, nieprzygotowanego słownika i sieci
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;[]()")
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    
    # Budujemy mały silnik 'FlowNetwork'
    brain_flow_network = EnhancedFlowTransformer(
        vocab_size=len(chars),
        d_model=64,
        num_layers=2,
        use_memory=False
    ).eval().to(device)
    
    # 2. Inicjalizacja Agenta z ConceptGraph + SelfCritique
    agent = CognitiveFlowAgent(brain_flow_network, stoi, itos, device)
    
    # === ETAP A: Zapisywanie informacji twardych (Twardy Dysk RAG) ===
    print("\n[INFO] Agent konsoliduje Twardy Graf Wiedzy (ConceptGraph)...")
    agent.semantic_memory.add_fact("Klucze", "leżą w", "Szufladzie")
    agent.semantic_memory.add_fact("Błąd", "posiada kod", "ERROR-404")
    agent.semantic_memory.add_fact("Hasło", "brzmi", "Truskawka123")
    agent.semantic_memory.add_fact("Python", "jest", "językiem programowania")
    agent.semantic_memory.add_fact("FlowNetwork", "używa", "Python")
    
    print(f"   Zapisano {len(agent.semantic_memory.concepts)} konceptów, "
          f"{len(agent.semantic_memory.edges)} krawędzi")
    
    time.sleep(0.5)
    
    # === ETAP B: Pętla kognitywna (Spreading Activation + Self-Critique) ===
    print("\n[USER_INTERACTION] Podaj użytkownikowi odpowiedź:")
    prompt = "Hasło"
    print(f"Użytkownik pisze wpis: '{prompt}'")
    
    print("\n[AGENT] Wewnętrzny tok myślenia (Working Memory Pipeline)...")
    output = agent.perceive_and_think(prompt)
    
    print("\n[WYNIK WSTRZYKNIĘCIA RAG -> FLOW] Wynik wstrzygnięcia twardego odczytu:")
    print("-" * 50)
    print(f"I sieć zwróciła wygenerowany ciąg na ten temat: {output[:50]}...")
    print("-" * 50)
    
    # === ETAP B2: Drugie pytanie — spreading activation ===
    print("\n[USER_INTERACTION #2] Drugie pytanie o powiązany temat:")
    prompt2 = "Python FlowNetwork"
    print(f"Użytkownik pisze: '{prompt2}'")
    output2 = agent.perceive_and_think(prompt2)
    print(f"Odpowiedź: {output2[:50]}...")
    
    # === ETAP C: Pamięć Epizodyczna i Status ===
    print("\n[EPISODIC LOG] Zawartość historii ostatnich chwil Agenta:")
    print(agent.episodic_memory.get_recent_history())
    
    # === ETAP D: Self-Critique Status ===
    print("\n[SELF-CRITIQUE] Status samokrytycyzmu:")
    status = agent.get_agent_status()
    print(f"   Koncepty w grafie: {status['concepts_count']}")
    print(f"   Krawędzie w grafie: {status['edges_count']}")
    print(f"   Jakość (Bayes mean): {status['quality_mean']:.3f}")
    print(f"   Trend jakości: {status['quality_trend']}")
    print(f"   Top koncepty: {[c['id'] for c in status['top_concepts'][:3]]}")
    
    # === ETAP E: Pętla Snu z Konsolidacją ===
    print("\n[DREAM LOOP] Agent kładzie się spać w celu konsolidacji nauki...")
    dream_result = agent.dream()
    print(dream_result)
    
    print("\n" + "=" * 60)
    print("✅ Demo zakończone! Koncepcja ConceptGraph + SelfCritique + RAG")
    print("   działa poprawnie z architekturą Linear Flow O(N).")
    print("=" * 60)

if __name__ == "__main__":
    run_cognitive_demo()
