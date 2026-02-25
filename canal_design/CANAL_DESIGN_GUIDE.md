# Canal Design Guide (Indian Standard Codes) 🇮🇳

This guide outlines the systematic engineering steps required for canal design in India, referencing the primary **Indian Standard (IS)** codes used at each stage.

## 📚 Key Reference Codes

| IS Code | Title | Purpose |
| :--- | :--- | :--- |
| **IS 5968:1987** | Guide for Planning and Layout of Canal Systems | Alignment, Curves, IPs |
| **IS 10430:2000** | Criteria for Design of Lined Canals | Lining, Freeboard, Section |
| **IS 7112:2002** | Design of Cross-Section for Unlined Canals | Alluvial Soil Design (Regime) |
| **IS 4839 (P1-3)** | Maintenance of Canals | Operations & Structural Safety |

---

## 🛠️ Step-by-Step Design Workflow

### Step 1: Layout & Alignment
**Standard:** **IS 5968:1987**
*   **Action:** Define the path of the canal based on topography.
*   **Critical Excerpt (Clause 5.2):** Alignment should be such that the canal commands maximum area with minimum length and avoids deep cutting or heavy filling.
*   **Curve Radius (Clause 8.1):** For unlined canals, the minimum radius must be followed based on discharge (Q).
    *   *Example:* For Q = 30 to 80 cumecs, Min Radius = **1000m**.
    *   *Lined Canal Note:* Radii can be reduced (typically 3x to 5x the bed width) as per **IS 10430**.

### Step 2: Determination of Cross-Section
**Standard:** **IS 10430:2000** (Lined) or **IS 7112:2002** (Unlined)
*   **Action:** Calculate Bed Width (B) and Depth (D).
*   **Logic (IS 10430):** Use Manning’s Formula $V = \frac{1}{n} R^{2/3} S^{1/2}$.
    *   **Side Slopes (Table 2):** 
        *   Concrete Lining: **1.5:1** (Standard).
        *   Stone Masonry: **1:1**.
*   **Critical Excerpt (Clause 4.2):** Velocity should be non-silting and non-scouring. For lined canals, max velocity is typically **2.5 m/s**.

### Step 3: Freeboard & Bank Design
**Standard:** **IS 10430:2000 (Table 1)**
*   **Action:** Add safety height above the Full Supply Level (FSL).
*   **Standard Excerpts:**
    *   Q < 0.75 cumecs: **0.30m** Freeboard.
    *   Q 0.75 to 1.5 cumecs: **0.50m** Freeboard.
    *   Q 1.5 to 85 cumecs: **0.60m** Freeboard.
    *   Q > 85 cumecs: **0.75m** Freeboard.

### Step 4: Lining Specifications
**Standard:** **IS 10430:2000 (Clause 5)**
*   **Action:** Select material and thickness.
*   **Criteria:** 
    *   Concrete (PCC): Standard thickness **100mm to 150mm** for large canals.
    *   Purpose: To minimize seepage losses (approx. 70-80% reduction).

---

## 🖥️ Implementation in `design_canal_is.py`

In our automated script, we followed these excerpts:
1.  **Alignment:** Used `get_tangent_points` to ensure smooth circular arcs as per **IS 5968**.
2.  **Section:** Applied the **1.5:1 side slope** and **0.6m freeboard** mandated by **IS 10430** for a 50 cumec discharge.
3.  **Visualization:** Added a concrete lining layer proxy ($T=100mm$) consistent with **IS 10430** structural guidelines.

---

## 📥 Official Download Links (Internet Archive / BIS)
*   [IS 5968:1987 - Canal Layout](https://archive.org/details/gov.in.is.5968.1987)
*   [IS 10430:2000 - Lined Canals](https://archive.org/details/gov.in.is.10430.2000)
*   [IS 7112:2002 - Unlined Alluvial Canals](https://archive.org/details/gov.in.is.7112.2002)
