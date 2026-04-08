# S.P.E.C.T.R.E

### *Self-Programming Environment for Complex Task Reconstruction & Evolution*

---

## What is S.P.E.C.T.R.E?

> **An OpenEnv-compliant environment where agents evolve from executors to system designers.**

S.P.E.C.T.R.E is a reinforcement learning environment built to push AI agents beyond simple step execution — into **self-programming intelligence**.

Most AI agents are executors—they follow instructions step-by-step, wasting compute and API tokens on repetitive loops.

SPECTRE transforms the agent from an executor into an Architect. Built on the OpenEnv framework, SPECTRE is a world-first RL environment that allows agents to perform Hierarchical Action Synthesis. Instead of just doing the work, the agent analyzes the task, builds its own "macros" (tools), and then recursively nests those tools to collapse complex 10-step data pipelines into a single, elegant execution.

---

## 🌍 Real-World Applications

- **Data Pipelines (ETL):** Automatically learns and compresses repetitive data workflows into reusable tools.  
- **Financial Reporting:** Builds macros for validation, aggregation, and revenue calculations across recurring reports.  
- **E-commerce Processing:** Optimizes bulk order handling through reusable pipeline patterns.  
- **Customer Support Automation:** Learns common workflows (refunds, issues) and reuses them efficiently.  
- **System Monitoring (Logs):** Detects repeated processing patterns and builds smarter alert pipelines.  

> **S.P.E.C.T.R.E is most powerful wherever workflows repeat — and efficiency comes from recognizing and reusing patterns.**

## Core Idea

Modern agents behave like function callers.

S.P.E.C.T.R.E challenges that.

> *"Why execute 10 steps manually when you can invent a tool that does it in 1?"*

Agents are rewarded for:

* ⚡ **Action compression**
* 🔁 **Tool reuse**
* 🧩 **Hierarchical abstraction**

This transforms the task from execution → into a **meta-reasoning problem**.

---

## Environment Design

### Primitive Operations

```
parse_data
validate_data
transform_data
export_result
```

These simulate real-world data pipeline workflows.

---

### 🛠️ Tool Creation (Self-Programming)

Agents dynamically define reusable tools:

```json
{
  "type": "create_tool",
  "name": "etl_batch",
  "sequence": ["parse_data", "validate_data", "transform_data"]
}
```

Then reuse them:

```json
{
  "type": "use_tool",
  "name": "etl_batch"
}
```

This enables:

* **Macro abstraction**
* **Hierarchical composition**
* **Exponential efficiency gains**

---

## Tasks

| Difficulty | Description                  | Focus                         |
| ---------- | ---------------------------- | ----------------------------- |
| 🟢 Easy    | Parse + validate two batches | Correct sequencing            |
| 🟡 Medium  | Full ETL ×2                  | Tool reuse                    |
| 🔴 Hard    | Full ETL ×3 + export         | Hierarchical self-programming |

---

## 🏆 Reward Philosophy

S.P.E.C.T.R.E uses a **shaped reward system** grounded in real-world optimization:

* **Incremental progress rewards**
* **Efficiency-based completion bonus**
* **Quality-aware scoring**
* **Compression bonus for tool usage**

> Agents are not just solving tasks — they are learning how to **optimize systems**.

---

## 📊 Output Metrics

Each episode tracks:

* ✔ **Progress completion**
* ✔ **Step efficiency**
* ✔ **Compression ratio**
* ✔ **Revenue generated**
* ✔ **Data quality score**

This enables **multi-dimensional evaluation**, beyond binary success.

---

## ⚙️ Tech Stack

```
FastAPI        → API layer (OpenEnv compliant)
Pydantic       → Typed schemas
Pandas         → Data pipeline simulation
Docker         → Containerized execution
Hugging Face   → Live deployment
```

---

## 🧪 Running Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Access API:

```
http://127.0.0.1:8000/docs
```

---

## Baseline Agent

```bash
python inference.py
```

Demonstrates:

* **Optimal strategies**
* **Tool creation behavior**
* **Reward-efficient execution**

---

## 🌐 Live Demo

👉 **Hugging Face Space**

```
https://huggingface.co/spaces/JeroFloyd/spectre-env
```

---

## 🧬 Why This Matters

S.P.E.C.T.R.E is not just another environment.

It represents a shift:

> **From task execution → to tool-building intelligence**

This is a step toward agents that:

* **Write their own abstractions**
* **Optimize workflows autonomously**
* **Scale reasoning through structure**

---

## Final Note

SPECTRE demonstrates that the future of Agentic AI isn't more parameters, it's better abstractions. By giving models the power to synthesize their own tools, we unlock a new level of compute efficiency and task complexity.

---

**Developed by: Jerovin Floyd**

**Institute: IIIT-Delhi (CSE)**

**Submission: OpenEnv Hackathon 2026 (Meta / Scaler / Hugging Face)**
