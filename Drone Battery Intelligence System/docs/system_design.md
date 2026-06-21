# System Design Document

## Predictive Range Monitoring System for UAVs

### 1. High-Level Architecture

```mermaid
flowchart TB
    subgraph DataIngestion["Data Ingestion Layer"]
        direction LR
        SENSORS[/"UAV Sensors<br/>(Battery, GPS, IMU)"/]
        ENV[/"Environmental Sensors<br/>(Weather, Terrain)"/]
        SIM[("Synthetic Data<br/>Generator")]
    end

    subgraph DataProcessing["Data Processing Layer"]
        direction TB
        CLEAN["Data Cleaning<br/>& Validation"]
        NORM["Normalization<br/>& Scaling"]
        FEAT["Feature<br/>Engineering"]
    end

    subgraph MLPipeline["ML Pipeline Layer"]
        direction TB
        
        subgraph Regression["Regression Module"]
            LR["Linear Regression<br/>from Scratch"]
            POLY["Polynomial Features"]
            RANGE["Range Estimator"]
        end
        
        subgraph Classification["Classification Module"]
            LOG["Logistic Regression<br/>from Scratch"]
            PROB["Probability Calibration"]
            SUCCESS["Success Predictor"]
        end
        
        subgraph TimeSeries["Time Series Module"]
            MA["Moving Average"]
            EXP["Exponential Smoothing"]
            FORECAST["SoC Forecaster"]
        end
        
        subgraph Recommender["Recommender Module"]
            PROFILE["Mission Profiles"]
            SIM_CALC["Similarity Calculator"]
            RECOMMEND["Route Recommender"]
        end
    end

    subgraph Serving["Model Serving Layer"]
        direction TB
        CACHE[("Model Cache")]
        INFERENCE["Inference Engine"]
        BATCH["Batch Predictions"]
        REALTIME["Real-time Predictions"]
    end

    subgraph Application["Application Layer"]
        direction LR
        STREAMLIT["Streamlit Dashboard"]
        ALERTS["Alert System"]
        LOGGING["Logging & Monitoring"]
    end

    SENSORS --> CLEAN
    ENV --> CLEAN
    SIM --> CLEAN
    
    CLEAN --> NORM
    NORM --> FEAT
    
    FEAT --> LR
    FEAT --> LOG
    FEAT --> MA
    FEAT --> PROFILE
    
    LR --> POLY --> RANGE
    LOG --> PROB --> SUCCESS
    MA --> EXP --> FORECAST
    PROFILE --> SIM_CALC --> RECOMMEND
    
    RANGE --> CACHE
    SUCCESS --> CACHE
    FORECAST --> CACHE
    RECOMMEND --> CACHE
    
    CACHE --> INFERENCE
    INFERENCE --> BATCH
    INFERENCE --> REALTIME
    
    BATCH --> STREAMLIT
    REALTIME --> STREAMLIT
    REALTIME --> ALERTS
    INFERENCE --> LOGGING
```

---

### 2. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input["Input Data"]
        B["Battery Data<br/>SoC, V, I, T"]
        E["Environmental<br/>Wind, Dust, Terrain"]
        F["Flight Data<br/>Distance, Speed"]
    end
    
    subgraph Features["Feature Engineering"]
        direction TB
        F1["Power Draw<br/>Features"]
        F2["Efficiency<br/>Metrics"]
        F3["Environmental<br/>Impact Scores"]
    end
    
    subgraph Models["ML Models"]
        M1["Range<br/>Estimator"]
        M2["Success<br/>Predictor"]
        M3["Time Series<br/>Forecaster"]
        M4["Route<br/>Recommender"]
    end
    
    subgraph Output["Predictions"]
        O1["Max Range (km)"]
        O2["Success Prob (%)"]
        O3["SoC Forecast"]
        O4["Optimal Routes"]
    end
    
    B --> F1 & F2
    E --> F2 & F3
    F --> F1 & F2
    
    F1 & F2 & F3 --> M1 --> O1
    F1 & F2 & F3 --> M2 --> O2
    F1 --> M3 --> O3
    F1 & F2 & F3 --> M4 --> O4
```

---

### 3. Component Details

#### 3.1 Data Generation Module

```mermaid
classDiagram
    class DataGenerator {
        +num_samples: int
        +time_range: tuple
        +generate_battery_data()
        +generate_environmental_data()
        +generate_flight_data()
        +add_noise()
        +save_to_csv()
    }
    
    class BatterySimulator {
        +initial_soc: float
        +discharge_rate: float
        +temperature_factor: float
        +simulate_discharge()
        +calculate_voltage()
        +calculate_current()
    }
    
    class EnvironmentSimulator {
        +terrain_types: list
        +wind_model: str
        +dust_model: str
        +simulate_conditions()
    }
    
    DataGenerator --> BatterySimulator
    DataGenerator --> EnvironmentSimulator
```

#### 3.2 ML Models (From Scratch)

```mermaid
classDiagram
    class LinearRegression {
        +weights: ndarray
        +bias: float
        +learning_rate: float
        +n_iterations: int
        +fit(X, y)
        +predict(X)
        +gradient_descent()
        +compute_cost()
        +get_r2_score()
    }
    
    class LogisticRegression {
        +weights: ndarray
        +bias: float
        +learning_rate: float
        +n_iterations: int
        +fit(X, y)
        +predict_proba(X)
        +predict(X)
        +sigmoid()
        +cross_entropy_loss()
        +get_auc()
    }
    
    class TimeSeriesModel {
        +window_size: int
        +alpha: float
        +moving_average()
        +exponential_smoothing()
        +forecast()
        +detect_anomalies()
    }
    
    class ContentRecommender {
        +mission_profiles: ndarray
        +similarity_matrix: ndarray
        +cosine_similarity()
        +get_recommendations()
        +rank_missions()
    }
```

---

### 4. Real-Time Pipeline

```mermaid
sequenceDiagram
    participant UAV as UAV Sensors
    participant Stream as Data Stream
    participant Preprocess as Preprocessor
    participant Models as ML Models
    participant Dashboard as Streamlit
    participant Alert as Alert System

    loop Every 100ms
        UAV->>Stream: Send telemetry
        Stream->>Preprocess: Buffer data
        Preprocess->>Preprocess: Clean & normalize
        Preprocess->>Models: Feature vector
        
        par Parallel Inference
            Models->>Models: Range estimation
            Models->>Models: Success prediction
            Models->>Models: SoC forecast
        end
        
        Models->>Dashboard: Update displays
        
        alt Low Battery/High Risk
            Models->>Alert: Trigger warning
            Alert->>UAV: Send command
        end
    end
```

---

### 5. Deployment Architecture

```mermaid
flowchart TB
    subgraph Development["Development Environment"]
        JUPYTER["Jupyter Notebook<br/>(Model Development)"]
        PYCHARM["PyCharm<br/>(Code Organization)"]
    end
    
    subgraph Artifacts["Model Artifacts"]
        WEIGHTS["Model Weights<br/>(NumPy .npy)"]
        PARAMS["Hyperparameters<br/>(JSON)"]
        SCALER["Scaler Params<br/>(JSON)"]
    end
    
    subgraph Deployment["Deployment"]
        STREAMLIT["Streamlit App"]
        
        subgraph Pages["Dashboard Pages"]
            P1["Range Estimator"]
            P2["Success Predictor"]
            P3["Live Monitoring"]
            P4["Recommendations"]
        end
    end
    
    JUPYTER --> WEIGHTS
    JUPYTER --> PARAMS
    JUPYTER --> SCALER
    PYCHARM --> STREAMLIT
    
    WEIGHTS --> STREAMLIT
    PARAMS --> STREAMLIT
    SCALER --> STREAMLIT
    
    STREAMLIT --> P1 & P2 & P3 & P4
```

---

### 6. Database Schema (For Logging)

```mermaid
erDiagram
    TELEMETRY {
        int id PK
        datetime timestamp
        float battery_soc
        float voltage
        float current
        float temperature
    }
    
    ENVIRONMENT {
        int id PK
        datetime timestamp
        string terrain_type
        float wind_speed
        float dust_level
    }
    
    PREDICTIONS {
        int id PK
        datetime timestamp
        float predicted_range
        float success_probability
        float soc_forecast
    }
    
    MISSIONS {
        int id PK
        string mission_type
        float distance
        float payload
        bool success
    }
    
    TELEMETRY ||--o{ PREDICTIONS : generates
    ENVIRONMENT ||--o{ PREDICTIONS : influences
    MISSIONS ||--o{ PREDICTIONS : evaluates
```

---

### 7. Monitoring & Alerting

```mermaid
flowchart LR
    subgraph Metrics["Key Metrics"]
        M1["Model Latency"]
        M2["Prediction Accuracy"]
        M3["Data Quality"]
        M4["System Health"]
    end
    
    subgraph Thresholds["Alert Thresholds"]
        T1["Latency > 100ms"]
        T2["Accuracy Drop > 5%"]
        T3["Missing Data > 1%"]
        T4["Low Battery < 20%"]
    end
    
    subgraph Actions["Alert Actions"]
        A1["Dashboard Warning"]
        A2["Model Retrain Flag"]
        A3["Data Pipeline Check"]
        A4["Emergency Landing"]
    end
    
    M1 --> T1 --> A1
    M2 --> T2 --> A2
    M3 --> T3 --> A3
    M4 --> T4 --> A4
```

---

### 8. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Processing | NumPy, Pandas | Data manipulation |
| ML Development | NumPy (from scratch) | Custom implementations |
| Visualization | Matplotlib, Seaborn | Charts & plots |
| Dashboard | Streamlit | Interactive UI |
| Development | Jupyter, PyCharm | Coding environments |
| Version Control | Git, GitHub | Source control |

---

### 9. Design Principles (Zinkevich's Rules)

| Principle | Implementation |
|-----------|----------------|
| **Rule #4**: Simple first model | Linear Regression baseline |
| **Rule #5**: Test infrastructure | Separate pipeline tests |
| **Rule #14**: Interpretable models | No black-box models initially |
| **Rule #29**: Train like serve | Same preprocessing code |
| **Rule #32**: Reuse code | Shared utility modules |
| **Rule #37**: Measure skew | Training/serving comparison |
