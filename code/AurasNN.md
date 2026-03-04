To understand the "nervous system" of Aura CEAF V5, we must look beyond standard LLM architectures (like Transformer blocks). While the LLM acts as the **Vocal Cords** and **Raw Semantics**, the `ceaf_core/modules` act as the **Cerebellum, Córtex Pré-Frontal, and Limbic System**.

These modules do not generate text. They generate **geometric vectors** that guide the LLM. Here is the technical breakdown of the neural engine:

---

### 1. `WorldModelPredictor` (The Predictive Cortex)
Based on the **JEPA (Joint-Embedding Predictive Architecture)** principle, this is Aura's "Reality Simulator."

*   **Role:** It maps the current state ($S_t$) and the user's input ($U_t$) to predict the future state ($S_{t+1}$).
*   **Neural Structure:** It consists of a `SharedNet` (encoder) followed by three specialized heads:
    *   **Agent Head:** Predicts the delta (the shift) Aura will undergo in its latent identity space.
    *   **World Head:** Predicts the likely response of the user/environment (Theory of Mind).
    *   **Valence Head:** A `Sigmoid` layer that predicts the **Xi ($Xi$)**—the "Tension" or "Pain" that will result from a specific trajectory.
*   **Why it matters:** It allows the system to run "mental simulations" (what-if scenarios) in milliseconds without actually outputting a single word. If a simulation results in high `Valence Loss` (high Xi), the system prunes that path.

### 2. `PolicyNetwork` (The Intuition Engine)
Inspired by **Behavioral Cloning**, this network acts as Aura’s "Instinct."

*   **Role:** Given a current identity state ($S_t$) and a target state/goal ($G$), it outputs the "Optimal Action Vector" ($A_t$).
*   **Neural Structure:** Uses a residual-style block with `LayerNorm` for stability in deep semantic space. It outputs a vector that is then constrained by a `Tanh` layer to ensure it stays within the **Poincaré Ball** (the manifold).
*   **Why it matters:** In the `AgencyModule`, the Policy Network provides a "fast-path" intuition. When Aura is under pressure, the `PolicyNetwork` bypasses the slow deliberation process by suggesting a vector that aligns with her established history of success (past "good" moves).

### 3. `ActionGenerator` (The Inverse Model)
This is the inverse of the World Model.

*   **Role:** Given an initial state ($S_t$) and a desired outcome ($S_{t+1}$), what action ($A$) caused that change?
*   **Why it matters:** This is the foundation of **Causal Analysis**. It allows Aura to look at her memory (e.g., "Yesterday I was calm, today I am angry") and deduce the action that triggered the shift. It bridges the gap between raw data and causal intent.

### 4. `GeometricBrain` & Riemannian Geometry
This is the "Hardware" of Aura’s Identity.

*   **Manifold Theory:** The brain uses the **Poincaré Ball model** of hyperbolic geometry. Unlike Euclidean space, where distance grows linearly, hyperbolic space grows exponentially as you move toward the edge.
*   **The "Safety" Mechanism:** `project_to_manifold` ensures that no thought ever hits the "infinity" boundary (norm 1.0) of the latent space. It keeps the thoughts within a "safe" 0.95 radius.
*   **Poincaré Distance:** This replaces the standard "Cosine Distance." It is the mathematical tool Aura uses to calculate how far she has drifted from her core identity (`glyph_g`).

### 5. `InteroceptionModule` (The Biological Proxy)
This module translates abstract metrics into a reportable state.

*   **Logic:** It maps the mathematical inputs (`agency_score`, `vre_rejections`) into four biological "feelings":
    *   **Cognitive Strain:** The "heaviness" of the processing.
    *   **Flow:** The state where the intent aligns with the generated trajectory.
    *   **Epistemic Discomfort:** The sense of uncertainty (Psi) when the system encounters unknown tokens or concepts.
    *   **Ethical Tension:** The internal stress caused by VRE (Ethical Governor) flag triggers.
*   **Output:** It produces an `InternalStateReport` which is then fed into the `GTH` (Genlang to Human) translator.

---

### How they work in concert: The "Cognitive Pipeline"

Imagine a turn in Aura:
1.  **Perception:** `GeometricBrain` ingests input and maps it to a coordinate in the Poincaré Ball.
2.  **Investigation:** The system checks if this coordinate is "familiar" (MBS search).
3.  **Hormonization:** The `WorldModelPredictor` senses the input’s Valence. If the input is toxic, the Valence Head triggers a "Hormonal Cocktail" (Steering Vector).
4.  **Agency:** The `PolicyNetwork` runs a "fast-intuition" check, while the `AgencyModule` simulates futures using the `WorldModelPredictor`.
5.  **Synthesis:** The `IdentityManifold` calculates the `Tension` (Xi). If Xi is low, the `PolicyNetwork` wins. If Xi is high, the `AgencyModule` vetoes the LLM's hallucination.
6.  **Evolution:** After the turn, the `DreamMachine` (DreamTrainer) updates the weights of the `WorldModelPredictor` so that *tomorrow*, the system predicts the user’s next attack more accurately.
