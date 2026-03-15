# 💻 Dynamic Pricer
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

A real-time quantitative finance web application built with Dash and Plotly. This tool allows users to dynamically price and structure complex financial derivatives (Partially Principal Protected Notes and Airbag Notes) by connecting directly to live market option chains.

## ✨ Key Features

* **Live Market Data Integration:** Fetches real-time stock prices, dividend yields, risk-free rates, and option chains (bid/ask/IV) via the Yahoo Finance API (`yfinance`).
* **Advanced Quantitative Engine:** Calculates theoretical option prices using the **Black-Scholes** model.
  * Simulates underlying asset trajectories using **Monte Carlo** simulations.
  * Solves for optimal commercial parameters based on a target banking margin.
* **Structured Products Supported:**
    * **PPPN (Partially Principal Protected Note):** Capital guarantee with upside participation.
    * **Airbag Note:** Optimized participation with downside protection up to a specific barrier.
* **Interactive 3D Risk Surfaces:** Visualizes Mark-to-Market (MtM) valuations across varying spot prices and time to maturity.
* **UI:** A dark-themed Glassmorphism interface built with Dash.

## 📈 Structured Products Supported

This application prices and structures two specific types of exotic financial derivatives, both dependent on the performance of a single underlying stock. Both products only generate a payoff at their maturity date.

### 1. Partially Principal Protected Note (PPPN)
The PPPN is designed for investors seeking a partial capital guarantee along with potential participation in significant market movements.
* **Capital Guarantee**: For an initial investment $N$ at time $t=0$, the investor is guaranteed to receive a fixed amount equal to 90% of $N$ at maturity.
* **Premium Mechanism**: On top of the 90% guarantee, the investor receives a premium based on the final stock price $S_T$ relative to two specific strike prices, $K_1$ and $K_2$ (where $K_1 \le S_0 \le K_2$).
* **Payoff Profile**: The premium is mathematically defined as:
  * If the stock drops significantly ($S_T < K_1$): `Premium` = $\frac{N}{S_0}(K_1 - S_T)$
  * If the stock stagnates ($K_1 \le S_T \le K_2$): `Premium` = 0
  * If the stock rallies ($S_T > K_2$): `Premium` = $\frac{N}{S_0}(S_T - K_2)$

### 2. Airbag Note
The Airbag Note provides no absolute principal protection but offers a structural buffer against moderate market downturns, combined with a tailored participation in market rallies.
* **Airbag Barrier**: The downside protection is active down to a specific "airbag" level, fixed at $A = 0.8 \cdot S_0$ (i.e., a 20% drop from the initial price).
* **Payoff Scenarios**: The final payoff is structured across three distinct market scenarios:
  * **Upside ($S_T \ge S_0$)**: The investor receives their initial investment back plus a partial participation rate ($p$) in the stock's positive performance: $N + p \cdot \frac{N}{S_0}(S_T - S_0)$
  * **Protected Downside ($A \le S_T < S_0$)**: The product's airbag deploys, providing full downside protection so the investor receives their initial capital $N$ back entirely.
  * **Unprotected Downside ($S_T < A$)**: If the final stock price breaches the airbag level, the protection is knocked out, and the investor takes a proportional loss: $\frac{N}{A} \cdot S_T$

## 📸 Screenshots
<img width="2559" height="838" alt="image" src="https://github.com/user-attachments/assets/5b2a7e9d-8bcd-4464-a861-e457d6d48c3e" />

<img width="2557" height="1179" alt="image" src="https://github.com/user-attachments/assets/627a8a62-1e6b-4076-82bf-b5f419ae0f70" />

<img width="2559" height="1178" alt="image" src="https://github.com/user-attachments/assets/e7e9e47a-56da-4fa5-94f3-62b977e1aec2" />


## 🚀 Installation & Setup

1. **Install the latest version of Python and the required packages:**
   ```
   python -m pip install dash dash-bootstrap-components plotly pandas numpy scipy yfinance
2. **run using this command in your Terminal:**
   ```
   python app.py
