import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta

# =============================================================================
# 0. NEW ART DIRECTION: GLASSMORPHISM
# =============================================================================
T_COLORS = {
    'bg_main': '#050B14',          
    'bg_panel': 'rgba(13, 22, 40, 0.6)', 
    'border_glow': 'rgba(0, 240, 255, 0.3)',
    'text_main': '#8AB4F8',        
    'text_kpi': '#FFFFFF',         
    'accent_cyan': '#00E5FF',      
    'accent_gold': '#FFD700',      
    'accent_green': '#00E676',     
    'accent_red': '#FF1744',       
    'grid_color': '#1A233A',
    'font': '"Segoe UI", "Roboto", "Helvetica Neue", sans-serif'
}

GLASS_STYLE = {
    'backgroundColor': T_COLORS['bg_panel'],
    'backdropFilter': 'blur(10px)',
    'border': f'1px solid {T_COLORS["border_glow"]}',
    'borderRadius': '12px',
    'boxShadow': '0 8px 32px 0 rgba(0, 0, 0, 0.37)'
}

# =============================================================================
# 1. FIXED GLOBAL PARAMETERS 
# =============================================================================
N = 1000              
# Note: target_margin and max_budget are now dynamically calculated in the callback

# Utility functions
def quant_reverse_engineer(payoff_points):
    S_pts, Y_pts = [p[0] for p in payoff_points], [p[1] for p in payoff_points]
    cash = Y_pts[0]
    stocks = (Y_pts[1] - Y_pts[0]) / (S_pts[1] - S_pts[0]) if len(S_pts) > 1 else 0
    options = []
    
    for i in range(1, len(payoff_points) - 1):
        K = S_pts[i]
        p_before = (Y_pts[i] - Y_pts[i-1]) / (S_pts[i] - S_pts[i-1])
        p_after = (Y_pts[i+1] - Y_pts[i]) / (S_pts[i+1] - S_pts[i])
        change = p_after - p_before
        if abs(change) > 1e-5: options.append({'Strike': K, 'Quantity': change})

    portfolio = {'cash': cash, 'stocks': stocks, 'puts': [], 'calls': []}
    for opt in options:
        K, Q = opt['Strike'], opt['Quantity']
        if portfolio['stocks'] > 0 and Q < 0 and abs(portfolio['stocks'] + Q) < 1e-5:
            portfolio['cash'] += portfolio['stocks'] * K
            portfolio['puts'].append({'Strike': K, 'Quantity': Q})
            portfolio['stocks'] = 0 
        elif portfolio['stocks'] < 0 and Q > 0 and abs(portfolio['stocks'] + Q) < 1e-5:
            portfolio['cash'] -= abs(portfolio['stocks']) * K
            portfolio['puts'].append({'Strike': K, 'Quantity': Q})
            portfolio['stocks'] = 0
        elif Q > 0: portfolio['calls'].append({'Strike': K, 'Quantity': Q})
        elif Q < 0: portfolio['calls'].append({'Strike': K, 'Quantity': Q})
    return portfolio

def eval_portfolio_MC(portfolio, ST_array):
    val = np.ones_like(ST_array) * portfolio['cash']
    val += portfolio['stocks'] * ST_array
    for call in portfolio['calls']: val += call['Quantity'] * np.maximum(ST_array - call['Strike'], 0)
    for put in portfolio['puts']: val += put['Quantity'] * np.maximum(put['Strike'] - ST_array, 0)
    return val

def calc_kpi(payoffs, N_val):
    returns = (payoffs - N_val) / N_val
    return np.mean(returns)*100, np.percentile(returns, 5)*100, np.mean(payoffs < N_val)*100

def bs_call(S, K, T, r, q, sigma):
    T = np.maximum(T, 1e-5)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, q, sigma):
    T = np.maximum(T, 1e-5)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

# --- RISK FUNCTIONS (ANALYTICAL GREEKS) ---
def bs_d1(S, K, T, r, q, sigma):
    T = np.maximum(T, 1e-5)
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def bs_delta(S, K, T, r, q, sigma, option_type='call'):
    d1 = bs_d1(S, K, T, r, q, sigma)
    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1)

def bs_gamma(S, K, T, r, q, sigma):
    d1 = bs_d1(S, K, T, r, q, sigma)
    return (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, q, sigma):
    # Vega expressed for a 1% change in volatility (divided by 100)
    d1 = bs_d1(S, K, T, r, q, sigma)
    return (S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)) / 100.0

def calc_portfolio_greeks(portfolio, S, T, r, q, sigma):
    delta = portfolio['stocks']
    gamma = 0.0
    vega = 0.0
    
    for call in portfolio['calls']:
        Q = call['Quantity']
        K = call['Strike']
        delta += Q * bs_delta(S, K, T, r, q, sigma, 'call')
        gamma += Q * bs_gamma(S, K, T, r, q, sigma)
        vega += Q * bs_vega(S, K, T, r, q, sigma)
        
    for put in portfolio['puts']:
        Q = put['Quantity']
        K = put['Strike']
        delta += Q * bs_delta(S, K, T, r, q, sigma, 'put')
        gamma += Q * bs_gamma(S, K, T, r, q, sigma)
        vega += Q * bs_vega(S, K, T, r, q, sigma)
        
    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega}

def eval_portfolio_3D(portfolio, S_m, T_m, r_val, q_val, sigma_val):
    MtM = portfolio['cash'] * np.exp(-r_val * T_m) 
    MtM += portfolio['stocks'] * S_m * np.exp(-q_val * T_m)
    for call in portfolio['calls']: MtM += call['Quantity'] * bs_call(S_m, call['Strike'], T_m, r_val, q_val, sigma_val)
    for put in portfolio['puts']: MtM += put['Quantity'] * bs_put(S_m, put['Strike'], T_m, r_val, q_val, sigma_val)
    return MtM

def apply_theme(fig, title):
    fig.update_layout(
        title={'text': title, 'font': {'color': T_COLORS['text_kpi'], 'size': 18, 'family': T_COLORS['font']}},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=T_COLORS['text_main'], family=T_COLORS['font']),
        xaxis=dict(showgrid=True, gridcolor=T_COLORS['grid_color'], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=T_COLORS['grid_color'], zeroline=True, zerolinecolor='gray'),
        margin=dict(l=30, r=30, t=50, b=30), hovermode="x unified"
    )
    return fig

# UI Components 
def kpi_card(title, value, color=T_COLORS['accent_cyan'], tooltip=""):
    # Create title block with info icon if tooltip is provided
    title_content = [html.Span(title)]
    if tooltip:
        title_content.append(html.Span(
            " ⓘ", 
            title=tooltip, # Native HTML attribute that creates the hover bubble
            style={'cursor': 'help', 'color': 'rgba(255,255,255,0.5)', 'fontSize': '13px', 'marginLeft': '4px'}
        ))
        
    return html.Div([
        html.Div(title_content, style={'fontSize': '11px', 'color': T_COLORS['text_main'], 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.Div(value, style={'fontSize': '24px', 'color': color, 'fontWeight': 'bold', 'marginTop': '8px'})
    ], style={'padding': '20px', 'borderRight': f'1px solid {T_COLORS["grid_color"]}', 'flex': '1', 'textAlign': 'center'})

def create_html_table(rows, color):
    header = html.Tr([
        html.Th("INSTRUMENT", style={'padding': '12px', 'textAlign': 'left', 'color': T_COLORS['text_main']}),
        html.Th("ACTION", style={'padding': '12px', 'textAlign': 'center', 'color': T_COLORS['text_main']}),
        html.Th("QUANTITY", style={'padding': '12px', 'textAlign': 'right', 'color': T_COLORS['text_main']}),
        html.Th("STRIKE", style={'padding': '12px', 'textAlign': 'right', 'color': T_COLORS['text_main']}),
        html.Th("UNIT PRICE", style={'padding': '12px', 'textAlign': 'right', 'color': T_COLORS['text_main']}),
        html.Th("TOTAL COST", style={'padding': '12px', 'textAlign': 'right', 'color': color})
    ], style={'borderBottom': f'2px solid {color}'})

    body_rows = []
    for row in rows:
        action_color = T_COLORS['accent_green'] if row['Action'] in ['SELL', 'PROFIT'] else T_COLORS['accent_red'] if row['Action'] == 'BUY' else T_COLORS['text_kpi']
        body_rows.append(html.Tr([
            html.Td(row['Instrument'], style={'padding': '10px', 'textAlign': 'left', 'fontWeight': 'bold', 'color': T_COLORS['text_kpi']}),
            html.Td(row['Action'], style={'padding': '10px', 'textAlign': 'center', 'color': action_color, 'fontWeight': 'bold'}),
            html.Td(row['Quantity'], style={'padding': '10px', 'textAlign': 'right', 'color': T_COLORS['text_kpi']}),
            html.Td(row['Strike'], style={'padding': '10px', 'textAlign': 'right', 'color': T_COLORS['text_kpi']}),
            html.Td(row['Unit_Price'], style={'padding': '10px', 'textAlign': 'right', 'color': T_COLORS['text_kpi']}),
            html.Td(row['Total'], style={'padding': '10px', 'textAlign': 'right', 'color': color, 'fontWeight': 'bold'})
        ], style={'borderBottom': f'1px solid {T_COLORS["grid_color"]}'}))

    return html.Table([html.Thead(header), html.Tbody(body_rows)], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '14px'})


# =============================================================================
# 5. USER INTERFACE GENERATION (DASH)
# =============================================================================
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': T_COLORS['bg_main'], 'fontFamily': T_COLORS['font'], 'padding': '40px', 'minHeight': '100vh'}, children=[
    
    html.H1("DYNAMIC PRICER", style={'color': T_COLORS['text_kpi'], 'textAlign': 'center', 'fontWeight': '300', 'letterSpacing': '4px', 'marginBottom': '40px'}),

    # --- REAL-TIME CONTROL PANEL ---
    html.Div(style={**GLASS_STYLE, 'marginBottom': '40px', 'padding': '30px', 'display': 'flex', 'gap': '30px', 'alignItems': 'flex-end'}, children=[
        html.Div([
            html.Label("UNDERLYING TICKER", style={'color': T_COLORS['text_main'], 'fontSize': '12px', 'letterSpacing': '1px'}),
            dcc.Input(id='ticker-input', type='text', value='PG', style={'width': '100%', 'height': '45px', 'fontSize': '16px', 'padding': '10px', 'backgroundColor': 'rgba(0,0,0,0.5)', 'color': T_COLORS['text_kpi'], 'border': f'1px solid {T_COLORS["grid_color"]}', 'borderRadius': '5px'})
        ], style={'flex': '1'}),
        
        html.Div([
            html.Label("TARGET MATURITY", style={'color': T_COLORS['text_main'], 'fontSize': '12px', 'letterSpacing': '1px'}),
            dcc.Input(id='date-input', type='text', placeholder='YYYY-MM-DD', style={'width': '100%', 'height': '45px', 'fontSize': '16px', 'padding': '10px', 'backgroundColor': 'rgba(0,0,0,0.5)', 'color': T_COLORS['text_kpi'], 'border': f'1px solid {T_COLORS["grid_color"]}', 'borderRadius': '5px'})
        ], style={'flex': '1'}),

        # --- NEW BOX: TARGET MARGIN ---
        html.Div([
            html.Label("TARGET MARGIN (%)", style={'color': T_COLORS['text_main'], 'fontSize': '12px', 'letterSpacing': '1px'}),
            dcc.Input(id='margin-input', type='number', value=2.0, min=0.0, max=100.0, step=0.1, style={'width': '100%', 'height': '45px', 'fontSize': '16px', 'padding': '10px', 'backgroundColor': 'rgba(0,0,0,0.5)', 'color': T_COLORS['text_kpi'], 'border': f'1px solid {T_COLORS["grid_color"]}', 'borderRadius': '5px'})
        ], style={'flex': '1'}),

        html.Button("PRICE IN REAL-TIME", id='compute-btn', n_clicks=0, style={'height': '45px', 'padding': '0 30px', 'backgroundColor': T_COLORS['accent_cyan'], 'color': '#000', 'border': 'none', 'borderRadius': '5px', 'fontWeight': 'bold', 'cursor': 'pointer', 'letterSpacing': '1px', 'flex': '0.5'})
    ]),
    
    # Potential error message
    html.Div(id='error-message', style={'color': T_COLORS['accent_red'], 'textAlign': 'center', 'marginBottom': '20px'}),

    # Results container to be populated by the callback
    dcc.Loading(
        id="loading-1",
        type="cube",
        color=T_COLORS['accent_cyan'],
        children=html.Div(id='dynamic-content')
    )
])


# =============================================================================
# 6. CALLBACK: QUANTITATIVE ENGINE CONNECTED TO YFINANCE
# =============================================================================
@app.callback(
    [Output('dynamic-content', 'children'),
     Output('error-message', 'children')],
    [Input('compute-btn', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('date-input', 'value'),
     State('margin-input', 'value')] # <-- Added State for margin
)
def update_dashboard(n_clicks, ticker_str, target_date_str, margin_val):
    if not ticker_str:
        return "", "Please enter a Ticker."
    
    try:
        # --- DYNAMIC MARGIN AND BUDGET MANAGEMENT ---
        try:
            target_margin = float(margin_val) / 100.0 if margin_val is not None else 0.02
        except ValueError:
            target_margin = 0.02 # Fallback to 2% in case of input error
            
        max_budget = N * (1 - target_margin)

        # 1. FETCH REAL-TIME DATA
        ticker = yf.Ticker(ticker_str.upper())
        hist = ticker.history(period="1d")
        if hist.empty:
            return "", f"Ticker {ticker_str} not found."
        S0 = hist['Close'].iloc[-1]
        
        # Fetch available market maturities
        available_dates = ticker.options
        if not available_dates:
            return "", f"No options available for {ticker_str}."
        
        # --- SMART DATE MANAGEMENT ---
        # 1. Define default target: Today + 365 days (T=1)
        target_date_default = datetime.now() + timedelta(days=365)
        
        # 2. Convert market dates to datetime objects
        available_datetimes = [datetime.strptime(d, "%Y-%m-%d") for d in available_dates]
        
        # 3. If the user entered a date, try to use it, else keep T=1
        if target_date_str:
            try:
                target_date = datetime.strptime(target_date_str.strip(), "%Y-%m-%d")
            except ValueError:
                target_date = target_date_default # Safety fallback
        else:
            target_date = target_date_default

        # 4. Find the REAL expiration date closest to our target
        closest_date_obj = min(available_datetimes, key=lambda d: abs(d - target_date))
        expiration_date = closest_date_obj.strftime("%Y-%m-%d")

        # Calculate T (Maturity in years)
        date_format = "%Y-%m-%d"
        T = (datetime.strptime(expiration_date, date_format) - datetime.now()).days / 365.25
        if T <= 0: T = 0.01

        # Risk-free rate (Approximation with US 3-Month Treasury ^IRX)
        try:
            irx = yf.Ticker("^IRX").history(period="1d")
            r = irx['Close'].iloc[-1] / 100.0 if not irx.empty else 0.04
        except:
            r = 0.04 # Fallback
            
        # Dividends
        q = ticker.info.get('dividendYield', 0.0)
        if q is None: 
            q = 0.0
        elif q > 0.20: # If yfinance acts up and returns > 20%
            q = q / 100.0 if q > 1 else 0.03 # Safety fallback

        # =============================================================================
        # --- NEW ROBUST PRICE MANAGEMENT (BID/ASK) WATERFALL ---
        # =============================================================================
        opt_chain = ticker.option_chain(expiration_date)
        calls = opt_chain.calls[['strike', 'bid', 'ask', 'impliedVolatility', 'lastPrice']].copy()
        puts = opt_chain.puts[['strike', 'bid', 'ask', 'impliedVolatility', 'lastPrice']].copy()

        # 1. Secure At-The-Money Implied Volatility (Sigma)
        atm_calls = calls[(calls['strike'] >= S0 * 0.9) & (calls['strike'] <= S0 * 1.1)]
        sigma = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else 0.20
        if pd.isna(sigma) or sigma <= 0.01: 
            sigma = 0.20

        # 2. Calculate THEORETICAL Black-Scholes price for ALL options (Vector)
        calls['BS_Price'] = bs_call(S0, calls['strike'], T, r, q, sigma)
        puts['BS_Price'] = bs_put(S0, puts['strike'], T, r, q, sigma)

        # 3. Fallback logic
        calls['Call_Ask'] = np.where(calls['ask'] > 0.01, calls['ask'], 
                                     np.where(calls['lastPrice'] > 0.01, calls['lastPrice'], calls['BS_Price']))
        
        calls['Call_Bid'] = np.where(calls['bid'] > 0.01, calls['bid'], 
                                     np.where(calls['lastPrice'] > 0.01, calls['lastPrice'], calls['BS_Price'] * 0.95))

        puts['Put_Ask'] = np.where(puts['ask'] > 0.01, puts['ask'], 
                                   np.where(puts['lastPrice'] > 0.01, puts['lastPrice'], puts['BS_Price']))
        
        puts['Put_Bid'] = np.where(puts['bid'] > 0.01, puts['bid'], 
                                   np.where(puts['lastPrice'] > 0.01, puts['lastPrice'], puts['BS_Price'] * 0.95))

        # 4. Clean up and merge for your solver
        calls = calls[['strike', 'Call_Bid', 'Call_Ask', 'impliedVolatility']]
        puts = puts[['strike', 'Put_Bid', 'Put_Ask']]

        options_data = pd.merge(puts, calls, on='strike', how='inner').rename(columns={'strike': 'Strike'})
        options_data.fillna(0.01, inplace=True) 
        
        if options_data.empty:
            return "", "The retrieved option chain is empty."

        # =============================================================================
        # 2. SOLVERS (DETERMINATION OF COMMERCIAL PARAMETERS) 
        # =============================================================================
        # --- PPPN ---
        protected_capital = 900
        zcb_cost_pppn = protected_capital * np.exp(-r * T) 
        budget_options_pppn = max_budget - zcb_cost_pppn
        ratio_pppn = N / S0

        best_k1, best_k2, min_cost_diff = S0, S0, float('inf') 
        put_ask_pppn, call_ask_pppn = 0, 0

        for _, row_put in options_data[options_data['Strike'] <= S0].iterrows():
            for _, row_call in options_data[options_data['Strike'] >= S0].iterrows():
                cost = ratio_pppn * (row_put['Put_Ask'] + row_call['Call_Ask'])
                if cost <= budget_options_pppn:
                    diff = budget_options_pppn - cost
                    if diff < min_cost_diff:
                        min_cost_diff = diff
                        best_k1, best_k2 = row_put['Strike'], row_call['Strike']
                        put_ask_pppn = row_put['Put_Ask']
                        call_ask_pppn = row_call['Call_Ask']

        total_cost_pppn = zcb_cost_pppn + ratio_pppn * (put_ask_pppn + call_ask_pppn)

        # --- Airbag ---
        A = 0.8 * S0
        zcb_cost_airbag = N * np.exp(-r * T) 
        
        closest_put_strike = options_data.iloc[(options_data['Strike'] - (S0 * 0.8)).abs().argsort()[:1]]['Strike'].values[0]
        closest_call_strike = options_data.iloc[(options_data['Strike'] - S0).abs().argsort()[:1]]['Strike'].values[0]

        put_bid_airbag = options_data.loc[options_data['Strike'] == closest_put_strike, 'Put_Bid'].values[0]
        call_ask_airbag = options_data.loc[options_data['Strike'] == closest_call_strike, 'Call_Ask'].values[0]
        
        put_quantity = N / A
        
        # --- AIRBAG CORRECTION ---
        call_cost = call_ask_airbag * (N / S0)
        budget_dispo_airbag = (put_quantity * put_bid_airbag) - zcb_cost_airbag + max_budget
        
        p_optimal_brut = budget_dispo_airbag / call_cost if call_cost > 0 else 0
        p_optimal = max(0.0, p_optimal_brut) 
        
        total_cost_airbag = zcb_cost_airbag - (put_quantity * put_bid_airbag) + (p_optimal * (N / S0) * call_ask_airbag)
        
        call_cost = call_ask_airbag * (N / S0)
        p_optimal = (put_quantity * put_bid_airbag - zcb_cost_airbag + max_budget) / call_cost if call_cost > 0 else 0
        total_cost_airbag = zcb_cost_airbag - (put_quantity * put_bid_airbag) + (p_optimal * (N / S0) * call_ask_airbag)

        # =============================================================================
        # 3. QUANTITATIVE ENGINE, SIMULATIONS & RISK MANAGEMENT
        # =============================================================================
        port_pppn = quant_reverse_engineer([(0, 900 + ratio_pppn * best_k1), (best_k1, 900), (best_k2, 900), (S0*2, 900 + ratio_pppn * (S0*2 - best_k2))])
        port_airbag = quant_reverse_engineer([(0, 0), (A, 1000), (S0, 1000), (S0*2, 1000 + p_optimal * (N/S0) * (S0*2 - S0))])

        # --- CALCULATE PORTFOLIO GREEKS ---
        greeks_pppn = calc_portfolio_greeks(port_pppn, S0, T, r, q, sigma)
        greeks_airbag = calc_portfolio_greeks(port_airbag, S0, T, r, q, sigma)
        
        # The Delta displayed at the desk represents the number of underlying shares 
        # the bank needs to sell (if Delta > 0) or buy (if Delta < 0) to be neutral.
        hedging_shares_pppn = -greeks_pppn['Delta']
        hedging_shares_airbag = -greeks_airbag['Delta']

        N_sim, steps = 100000, 252 
        dt = T / steps
        np.random.seed(42)
        Z = np.random.standard_normal((steps, N_sim))
        S = np.zeros((steps + 1, N_sim))
        S[0] = S0
        drift = r - q - 0.5 * sigma**2 
        for i in range(steps): 
            S[i+1] = S[i] * np.exp(drift * dt + sigma * np.sqrt(dt) * Z[i])
        ST_MC = S[-1]

        payoff_pppn_mc = eval_portfolio_MC(port_pppn, ST_MC)
        payoff_airbag_mc = eval_portfolio_MC(port_airbag, ST_MC)

        kpi_p = calc_kpi(payoff_pppn_mc, N)
        kpi_a = calc_kpi(payoff_airbag_mc, N)

        # --- 3D Pricing ---
        S_grid, T_grid = np.linspace(S0*0.5, S0*1.5, 40), np.linspace(T, 0.0, 40)
        S_mesh, T_mesh = np.meshgrid(S_grid, T_grid)

        PnL_PPPN_3D = eval_portfolio_3D(port_pppn, S_mesh, T_mesh, r, q, sigma) - N
        PnL_Airbag_3D = eval_portfolio_3D(port_airbag, S_mesh, T_mesh, r, q, sigma) - N

        # =============================================================================
        # 4. PLOTLY CHARTS
        # =============================================================================
        ST_range = np.linspace(S0*0.5, S0*1.8, 400)
        pnl_stock = ((N / S0) * ST_range) - N

        fig_paths = go.Figure()
        for i in range(min(170, N_sim)): 
            fig_paths.add_trace(go.Scatter(y=S[:, i], mode='lines', line=dict(width=1, color='rgba(0, 229, 255, 0.16)'), showlegend=False))
        fig_paths.add_trace(go.Scatter(y=np.mean(S, axis=1), mode='lines', name='Average', line=dict(width=3, color=T_COLORS['accent_cyan'])))
        fig_paths = apply_theme(fig_paths, f"UNDERLYING TRAJECTORIES ({ticker_str.upper()}) - MONTE CARLO")

        def create_pnl_chart(data_product, title, color):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ST_range, y=data_product, name='Product', line=dict(color=color, width=3)))
            fig.add_trace(go.Scatter(x=ST_range, y=pnl_stock, name=f'Buy Stock {ticker_str.upper()}', line=dict(color=T_COLORS['text_main'], width=2, dash='dot')))
            fig.add_vline(x=S0, line_dash="dash", line_color="gray")
            return apply_theme(fig, title)

        fig_pnl_pppn = create_pnl_chart(eval_portfolio_MC(port_pppn, ST_range) - N, "THEORETICAL P&L", T_COLORS['accent_cyan'])
        fig_pnl_airbag = create_pnl_chart(eval_portfolio_MC(port_airbag, ST_range) - N, "THEORETICAL P&L", T_COLORS['accent_gold'])

        def create_3d(Z_data, color_scale, title):
            fig = go.Figure(data=[go.Surface(z=Z_data, x=S_grid, y=T_grid, colorscale=color_scale, opacity=0.9)])
            fig.update_layout(
                title={'text': title, 'font': {'color': T_COLORS['text_kpi'], 'family': T_COLORS['font']}},
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=T_COLORS['text_main']),
                scene=dict(
                    xaxis=dict(title='Underlying Price ($)', gridcolor=T_COLORS['grid_color'], backgroundcolor='rgba(0,0,0,0)'),
                    yaxis=dict(title='Time (Years)', gridcolor=T_COLORS['grid_color'], backgroundcolor='rgba(0,0,0,0)'),
                    zaxis=dict(title='MtM Valuation ($)', gridcolor=T_COLORS['grid_color'], backgroundcolor='rgba(0,0,0,0)')
                ), margin=dict(l=0, r=0, b=0, t=40)
            )
            return fig

        fig_3d_pppn = create_3d(PnL_PPPN_3D, 'Tealgrn', "3D RISK SURFACE (MtM)")
        fig_3d_airbag = create_3d(PnL_Airbag_3D, 'Plasma', "3D RISK SURFACE (MtM)")

        # =============================================================================
        # RESULT LAYOUT
        # =============================================================================
        table_data_pppn = [
            {'Instrument': 'Zero-Coupon Bond', 'Action': 'BUY', 'Quantity': '-', 'Strike': '-', 'Unit_Price': '-', 'Total': f"- ${zcb_cost_pppn:.2f}"},
            {'Instrument': 'Put Option', 'Action': 'BUY', 'Quantity': f"{ratio_pppn:.4f}", 'Strike': f"${best_k1:.2f}", 'Unit_Price': f"${put_ask_pppn:.2f}", 'Total': f"- ${ratio_pppn * put_ask_pppn:.2f}"},
            {'Instrument': 'Call Option', 'Action': 'BUY', 'Quantity': f"{ratio_pppn:.4f}", 'Strike': f"${best_k2:.2f}", 'Unit_Price': f"${call_ask_pppn:.2f}", 'Total': f"- ${ratio_pppn * call_ask_pppn:.2f}"},
            {'Instrument': 'MANUFACTURING BALANCE', 'Action': '-', 'Quantity': '-', 'Strike': '-', 'Unit_Price': 'Total Cost:', 'Total': f"- ${total_cost_pppn:.2f}"},
            {'Instrument': 'BANK MARGIN', 'Action': 'PROFIT', 'Quantity': '-', 'Strike': '-', 'Unit_Price': f"Sold: ${N:.2f}", 'Total': f"+ ${N - total_cost_pppn:.2f}"}
        ]

        table_data_airbag = [
            {'Instrument': 'Zero-Coupon Bond', 'Action': 'BUY', 'Quantity': '-', 'Strike': '-', 'Unit_Price': '-', 'Total': f"- ${zcb_cost_airbag:.2f}"},
            {'Instrument': 'Put Option (Funding)', 'Action': 'SELL', 'Quantity': f"{put_quantity:.4f}", 'Strike': f"${closest_put_strike:.2f}", 'Unit_Price': f"${put_bid_airbag:.2f}", 'Total': f"+ ${put_quantity * put_bid_airbag:.2f}"},
            {'Instrument': 'Call Option (Participation)', 'Action': 'BUY', 'Quantity': f"{p_optimal * N / S0:.4f}", 'Strike': f"${closest_call_strike:.2f}", 'Unit_Price': f"${call_ask_airbag:.2f}", 'Total': f"- ${p_optimal * (N/S0) * call_ask_airbag:.2f}"},
            {'Instrument': 'MANUFACTURING BALANCE', 'Action': '-', 'Quantity': '-', 'Strike': '-', 'Unit_Price': 'Total Cost:', 'Total': f"- ${total_cost_airbag:.2f}"},
            {'Instrument': 'BANK MARGIN', 'Action': 'PROFIT', 'Quantity': '-', 'Strike': '-', 'Unit_Price': f"Sold: ${N:.2f}", 'Total': f"+ ${N - total_cost_airbag:.2f}"}
        ]

        content = html.Div([
            
            # Displaying Live Market Data
            html.Div(style={**GLASS_STYLE, 'marginBottom': '40px', 'padding': '20px', 'display': 'flex', 'justifyContent': 'space-around'}, children=[
                html.Div(f"S0: ${S0:.2f}", style={'color': T_COLORS['accent_cyan'], 'fontWeight': 'bold'}),
                html.Div(f"Maturity: {expiration_date} (T={T:.2f} years)", style={'color': T_COLORS['text_kpi']}),
                html.Div(f"IV (Sigma): {sigma*100:.1f} %", style={'color': T_COLORS['accent_gold']}),
                html.Div(f"Risk-free rate: {r*100:.2f} %", style={'color': T_COLORS['text_kpi']}),
                html.Div(f"Dividend: {q*100:.2f} %", style={'color': T_COLORS['text_kpi']}),
            ]),

            html.Div([dcc.Graph(figure=fig_paths)], style={**GLASS_STYLE, 'marginBottom': '40px', 'padding': '20px'}),

            # =========================================================================
            # BLOCK 1 : PPPN
            # =========================================================================
            html.Div(style={**GLASS_STYLE, 'marginBottom': '50px', 'padding': '30px'}, children=[
                html.H2("PARTIALLY PRINCIPAL PROTECTED NOTE (PPPN)", style={'color': T_COLORS['accent_cyan'], 'marginTop': '0', 'letterSpacing': '2px', 'borderBottom': f'1px solid {T_COLORS["grid_color"]}', 'paddingBottom': '15px'}),
                
                html.Div([
                    kpi_card("Selling Price", f"${N}", T_COLORS['text_kpi']),
                    kpi_card("Capital Guarantee", f"${protected_capital}", T_COLORS['accent_cyan']),
                    kpi_card("Expected Return", f"{kpi_p[0]:.2f} %", T_COLORS['accent_green']),
                    kpi_card("Loss Risk (<$1000)", f"{kpi_p[2]:.2f} %", T_COLORS['accent_red']),
                    kpi_card("Net Margin", f"${N - total_cost_pppn:.2f}", T_COLORS['accent_green'])
                ], style={'display': 'flex', 'backgroundColor': 'rgba(0,0,0,0.2)', 'borderRadius': '8px', 'marginBottom': '30px'}),
                
                html.H3("STRUCTURING DETAILS (TERM SHEET)", style={'color': T_COLORS['text_main'], 'fontSize': '14px', 'letterSpacing': '1px'}),
                html.Div(create_html_table(table_data_pppn, T_COLORS['accent_cyan']), style={'backgroundColor': 'rgba(0,0,0,0.2)', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '30px'}),

                html.Div([
                    html.Div([dcc.Graph(figure=fig_pnl_pppn)], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([dcc.Graph(figure=fig_3d_pppn, style={'height': '500px'})], style={'width': '60%', 'display': 'inline-block'}),
                ], style={'marginBottom': '30px'}),

                # Risk monitor integration INSIDE the PPPN block
                html.H3("RISK MONITOR (GREEKS)", style={'color': T_COLORS['text_main'], 'fontSize': '14px', 'letterSpacing': '1px'}),
                html.Div([
                    kpi_card("Δ Delta (Global Pos.)", f"{greeks_pppn['Delta']:.2f}", T_COLORS['text_kpi'], 
                             tooltip="The stock equivalent of your portfolio. A positive Delta means you profit if the market goes up."),
                    
                    kpi_card("Action Hedging (Shares)", f"{hedging_shares_pppn:.2f}", T_COLORS['accent_green'] if hedging_shares_pppn > 0 else T_COLORS['accent_red'], 
                             tooltip="Immediate desk action: Exact quantity of underlying shares to buy (green) or sell (red) to immunize the portfolio against small market movements."),
                    
                    kpi_card("Γ Gamma", f"{greeks_pppn['Gamma']:.4f}", T_COLORS['accent_cyan'], 
                             tooltip="Measures the rate of change of your Delta. A high Gamma means your position is unstable and will require very frequent hedging readjustments."),
                    
                    kpi_card("V Vega (per 1% IV)", f"${greeks_pppn['Vega']:.2f}", T_COLORS['accent_cyan'], 
                             tooltip="The direct dollar impact on your P&L if the market's implied volatility (IV) suddenly increases by 1 point (e.g., from 20% to 21%).")
                ], style={'display': 'flex', 'backgroundColor': 'rgba(13, 22, 40, 0.8)', 'border': f'1px solid {T_COLORS["grid_color"]}', 'borderRadius': '8px'})
            ]),

            # =========================================================================
            # BLOCK 2 : AIRBAG
            # =========================================================================
            html.Div(style={**GLASS_STYLE, 'marginBottom': '50px', 'padding': '30px'}, children=[
                html.H2("AIRBAG NOTE (OPTIMIZED PARTICIPATION)", style={'color': T_COLORS['accent_gold'], 'marginTop': '0', 'letterSpacing': '2px', 'borderBottom': f'1px solid {T_COLORS["grid_color"]}', 'paddingBottom': '15px'}),
                
                html.Div([
                    kpi_card("Selling Price", f"${N}", T_COLORS['text_kpi']),
                    kpi_card("Airbag Level (A)", f"${A:.2f}", T_COLORS['accent_gold']),
                    kpi_card("Participation (p)", f"{p_optimal*100:.2f} %", T_COLORS['accent_gold']),
                    kpi_card("Loss Risk (<$1000)", f"{kpi_a[2]:.2f} %", T_COLORS['accent_red']),
                    kpi_card("Net Margin", f"${N - total_cost_airbag:.2f}", T_COLORS['accent_green'])
                ], style={'display': 'flex', 'backgroundColor': 'rgba(0,0,0,0.2)', 'borderRadius': '8px', 'marginBottom': '30px'}),

                html.H3("STRUCTURING DETAILS (TERM SHEET)", style={'color': T_COLORS['text_main'], 'fontSize': '14px', 'letterSpacing': '1px'}),
                html.Div(create_html_table(table_data_airbag, T_COLORS['accent_gold']), style={'backgroundColor': 'rgba(0,0,0,0.2)', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '30px'}),

                html.Div([
                    html.Div([dcc.Graph(figure=fig_pnl_airbag)], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([dcc.Graph(figure=fig_3d_airbag, style={'height': '500px'})], style={'width': '60%', 'display': 'inline-block'}),
                ], style={'marginBottom': '30px'}),

                # Risk monitor integration INSIDE the Airbag block
                html.H3("RISK MONITOR (GREEKS)", style={'color': T_COLORS['text_main'], 'fontSize': '14px', 'letterSpacing': '1px'}),
                html.Div([
                    kpi_card("Δ Delta (Global Pos.)", f"{greeks_airbag['Delta']:.2f}", T_COLORS['text_kpi'], 
                             tooltip="The stock equivalent of your portfolio. A positive Delta means you profit if the market goes up."),
                    
                    kpi_card("Action Hedging (Shares)", f"{hedging_shares_airbag:.2f}", T_COLORS['accent_green'] if hedging_shares_airbag > 0 else T_COLORS['accent_red'], 
                             tooltip="Immediate desk action: Exact quantity of underlying shares to buy (green) or sell (red) to immunize the portfolio against small market movements."),
                    
                    kpi_card("Γ Gamma", f"{greeks_airbag['Gamma']:.4f}", T_COLORS['accent_gold'], 
                             tooltip="Measures the rate of change of your Delta. A high Gamma means your position is unstable and will require very frequent hedging readjustments."),
                    
                    kpi_card("V Vega (per 1% IV)", f"${greeks_airbag['Vega']:.2f}", T_COLORS['accent_gold'], 
                             tooltip="The direct dollar impact on your P&L if the market's implied volatility (IV) suddenly increases by 1 point (e.g., from 20% to 21%).")
                ], style={'display': 'flex', 'backgroundColor': 'rgba(13, 22, 40, 0.8)', 'border': f'1px solid {T_COLORS["grid_color"]}', 'borderRadius': '8px'})
            ])
        ])

        return content, ""

    except Exception as e:
        return "", f"Error during processing: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)