# Architecture Visual Comparison

## LSTM vs S4D Data Flow

```
═══════════════════════════════════════════════════════════════════════════════
                              LSTM ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

Input Batch: (32, 240, 15)
    |
    | 32 = batch size
    | 240 = sequence length (60 seconds at 4Hz)
    | 15 = features (3 GND + 6 GS13 + 6 CPS channels)
    |
    ├─────────────────────────────────────────────────────────────┐
    │                      LSTM Layer 1                           │
    │  hidden_size = 128, bidirectional = False                   │
    │  Input: (32, 240, 15) → Output: (32, 240, 128)             │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ├─────────────────────────────────────────────────────────────┐
    │                      LSTM Layer 2                           │
    │  + Dropout (0.2)                                            │
    │  Input: (32, 240, 128) → Output: (32, 240, 128)            │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ├─────────────────────────────────────────────────────────────┐
    │                      LSTM Layer 3                           │
    │  + Dropout (0.2)                                            │
    │  Input: (32, 240, 128) → Output: (32, 240, 128)            │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ├─────────────────────────────────────────────────────────────┐
    │               Take Last Timestep [:, -1, :]                 │
    │  Input: (32, 240, 128) → Output: (32, 128)                 │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ├─────────────────────────────────────────────────────────────┐
    │                    Linear Layer (FC)                        │
    │  Input: (32, 128) → Output: (32, 6)                        │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
Output: (32, 6)
    |
    | 6 predictions for GS13 channels:
    | [GS13X, GS13Y, GS13Z, GS13RX, GS13RY, GS13RZ]
    |

Parameters: ~300K
Complexity: O(N) per layer, sequential
Memory: High (stores all hidden states)


═══════════════════════════════════════════════════════════════════════════════
                               S4D ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

Input Batch: (32, 240, 15)
    |
    | 32 = batch size
    | 240 = sequence length (60 seconds at 4Hz)
    | 15 = features (3 GND + 6 GS13 + 6 CPS channels)
    |
    ├─────────────────────────────────────────────────────────────┐
    │              Input Projection (Linear)                      │
    │  Map features to model dimension                            │
    │  Input: (32, 240, 15) → Output: (32, 240, 128)             │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ├─────────────────────────────────────────────────────────────┐
    │           Transpose for S4D Processing                      │
    │  Rearrange to (batch, channels, length)                     │
    │  Input: (32, 240, 128) → Output: (32, 128, 240)            │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                      S4D Block 1                            │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │  Pre/Post LayerNorm                                   │ │
    │  │  S4D Layer (d_model=128, d_state=64)                  │ │
    │  │  - State Space: dx/dt = Ax + Bu, y = Cx + Du         │ │
    │  │  - Diagonal A matrix for efficiency                   │ │
    │  │  - FFT-based convolution O(N log N)                   │ │
    │  │  Dropout (0.1)                                        │ │
    │  │  + Residual Connection                                │ │
    │  └───────────────────────────────────────────────────────┘ │
    │  Input: (32, 128, 240) → Output: (32, 128, 240)            │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                      S4D Block 2                            │
    │  [Same structure as Block 1]                                │
    │  Input: (32, 128, 240) → Output: (32, 128, 240)            │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                      S4D Block 3                            │
    │  [Same structure as Block 1]                                │
    │  Input: (32, 128, 240) → Output: (32, 128, 240)            │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                      S4D Block 4                            │
    │  [Same structure as Block 1]                                │
    │  Input: (32, 128, 240) → Output: (32, 128, 240)            │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ├─────────────────────────────────────────────────────────────┐
    │           Transpose Back from S4D Format                    │
    │  Rearrange to (batch, length, channels)                     │
    │  Input: (32, 128, 240) → Output: (32, 240, 128)            │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ├─────────────────────────────────────────────────────────────┐
    │               Take Last Timestep [:, -1, :]                 │
    │  Input: (32, 240, 128) → Output: (32, 128)                 │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ├─────────────────────────────────────────────────────────────┐
    │             Output Projection (Linear)                      │
    │  Input: (32, 128) → Output: (32, 6)                        │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
Output: (32, 6)
    |
    | 6 predictions for GS13 channels:
    | [GS13X, GS13Y, GS13Z, GS13RX, GS13RY, GS13RZ]
    |

Parameters: ~250K
Complexity: O(N log N) per layer, parallelizable
Memory: Medium (convolution-based, no hidden state storage)


═══════════════════════════════════════════════════════════════════════════════
                          STATE SPACE MODEL DETAILS
═══════════════════════════════════════════════════════════════════════════════

S4D uses continuous-time state space equations:

    dx/dt = A·x(t) + B·u(t)    ← State evolution
    y(t) = C·x(t) + D·u(t)     ← Output equation

Where:
    x(t) ∈ ℝᴺ  : Hidden state (N = d_state = 64)
    u(t) ∈ ℝᴴ  : Input (H = d_model = 128)
    y(t) ∈ ℝᴴ  : Output (H = d_model = 128)
    
    A ∈ ℝᴺˣᴺ  : State transition matrix (DIAGONAL!)
    B ∈ ℝᴺˣᴴ  : Input projection
    C ∈ ℝᴴˣᴺ  : Output projection
    D ∈ ℝᴴ    : Skip connection

Discretization (via bilinear transform):
    
    x[k+1] = Ā·x[k] + B̄·u[k]
    y[k] = C·x[k] + D·u[k]
    
    where: Ā = (I - Δt/2·A)⁻¹(I + Δt/2·A)
           B̄ = (I - Δt/2·A)⁻¹·Δt·B

Convolution kernel:
    
    K[0] = C·B̄
    K[i] = C·Āⁱ⁻¹·B̄  for i > 0
    
    Then: y = u ⊛ K  (convolution, computed via FFT)


═══════════════════════════════════════════════════════════════════════════════
                              KEY DIFFERENCES
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────┬──────────────────────┬──────────────────────────────┐
│    Aspect       │        LSTM          │            S4D               │
├─────────────────┼──────────────────────┼──────────────────────────────┤
│ Computation     │ Sequential O(N)      │ Parallel O(N log N)          │
│ Memory          │ Stores hidden states │ Convolution-based            │
│ Long-range      │ Weak (>200 steps)    │ Strong (1000+ steps)         │
│ Training speed  │ Fast convergence     │ Slower, more stable          │
│ Parameters      │ ~300K                │ ~250K                        │
│ Parallelization │ Limited (sequential) │ High (FFT-based)             │
│ Gradient flow   │ Can vanish           │ Better structured            │
└─────────────────┴──────────────────────┴──────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                          CAUSALITY & PREDICTION
═══════════════════════════════════════════════════════════════════════════════

Both models are strictly CAUSAL (no future information):

Timeline:
    
    ├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
    t=0   t=1   t=2  ...  t=237 t=238 t=239      t=243
    │◄──────────── 240 timesteps ──────────►│      │
    │           (60 seconds @ 4Hz)          │      │
    │                                       │      │
    │          Model processes              │      ▼
    │          this sequence                │   Prediction
    │                                       │   (1 second ahead)
    │                                       │
    │◄───────────── INPUT ─────────────────►│
                                            ▲
                                    Take last timestep
                                    for prediction

Both models:
1. Process 240 timesteps (60 seconds)
2. Take ONLY the last timestep's representation
3. Project to 6 outputs (GS13 channels)
4. Predict values at t=243 (1 second in future)

No future information leaks because:
- LSTM: Sequential processing is inherently causal
- S4D: Convolution kernel is causal by construction
```