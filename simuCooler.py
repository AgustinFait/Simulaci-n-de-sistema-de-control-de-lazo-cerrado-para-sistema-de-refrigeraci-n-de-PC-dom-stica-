import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches

# ==========================================
# CONFIGURACIÓN
# ==========================================

TIEMPO_VENTANA = 150   # segundos visibles en pantalla
INTERVAL_MS    = 120   # ms entre pasos

np.random.seed(None)

# ==========================================
# ESTADO DE SIMULACIÓN
# ==========================================

state = {
    't':             0,
    'T':             35.0,
    'T_ant':         35.0,
    'PWM':           50.0,
    'running':       False,
    'timer':         None,
    # historia
    'tArr':          [],
    'tempArr':       [],
    'refArr':        [],
    'errArr':        [],
    'pertArr':       [],
    'pwmArr':        [],
    # carga aleatoria
    'cpu_actual':    20.0,
    'cpu_target':    20.0,
    'perturb':       0,
    'perturb_timer': 0,
    'perturb_cd':    0,
    # falla
    'fault':      False,
    'fault_msg':  '',
}

PERTURB_LABELS = ['Normal', 'Uso de GPU/RAM elevado', 'Embalamiento termico', 'Polvo', 'Ruido eléctrico']
PERTURB_COLORS = ['#aaaaaa', '#2a78d6', '#e34948', '#eda100', '#1baf7a']

# ==========================================
# FIGURA
# ==========================================

fig = plt.figure(figsize=(17, 10))
fig.patch.set_facecolor('#f4f3ee')
plt.suptitle('Control de Refrigeración PC — Simulación en tiempo real',
             fontsize=12, fontweight='bold', color='#222', y=0.99)

gs_main = gridspec.GridSpec(4, 1, left=0.33, right=0.97,
                             top=0.94, bottom=0.05, hspace=0.55)

ax0 = fig.add_subplot(gs_main[0])
ax1 = fig.add_subplot(gs_main[1])
ax2 = fig.add_subplot(gs_main[2])
ax3 = fig.add_subplot(gs_main[3])

CHART_AXES = [ax0, ax1, ax2, ax3]
for ax in CHART_AXES:
    ax.set_facecolor('#ffffff')
    ax.grid(True, color='#e0e0e0', linewidth=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')

ax0.set_title('θi(t) — Temperatura de referencia', fontsize=9, color='#444', pad=3)
ax1.set_title('θo(t) — Temperatura real CPU', fontsize=9, color='#444', pad=3)
ax2.set_title('e(t) — Señal de error  (θi − θo)', fontsize=9, color='#444', pad=3)
ax3.set_title('CPU % / Perturbación activa', fontsize=9, color='#444', pad=3)
ax3.set_xlabel('Tiempo (s)', fontsize=8)

for ax in CHART_AXES:
    ax.tick_params(labelsize=8)

# líneas
line_ref,  = ax0.plot([], [], color='#2a78d6', lw=2, linestyle='--')
line_temp, = ax1.plot([], [], color='#e34948', lw=2, label='θo(t)')
line_ref2, = ax1.plot([], [], color='#2a78d6', lw=1.2, linestyle='--', alpha=0.5, label='θi(t)')
ax1.legend(fontsize=8, loc='upper right')
line_err,  = ax2.plot([], [], color='#4a3aa7', lw=2)
ax2.axhline(0, color='#999', lw=0.8, linestyle='--')
line_cpu,  = ax3.plot([], [], color='#e34948', lw=1.5, label='CPU %', zorder=3)
bar_holder = {'bars': None}

# anotación perturbación activa
ann_perturb = ax1.annotate('', xy=(0.98, 0.92), xycoords='axes fraction',
                            fontsize=8, ha='right', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', fc='#fff3cd', ec='#eda100', lw=1))

# ==========================================
# PANEL IZQUIERDO — SLIDERS
# ==========================================

PANEL_FC = '#ebebE3'

def section_label(y, text):
    fig.text(0.01, y, text, fontsize=7, color='#888', va='top',
             style='italic', fontweight='bold')

def add_slider(label, desc, ypos, vmin, vmax, vinit, vstep, color='#2a78d6'):
    ax_sl = fig.add_axes([0.015, ypos, 0.275, 0.025], facecolor=PANEL_FC)
    sl = Slider(ax_sl, '', vmin, vmax, valinit=vinit, valstep=vstep, color=color)
    sl.valtext.set_fontsize(8)
    sl.valtext.set_color('#222')
    fig.text(0.015, ypos + 0.028, f'{label}   [{vmin}–{vmax}]   →  {desc}',
             fontsize=7.5, color='#333', va='bottom')
    return sl

# --- Controlador ---
section_label(0.945, 'CONTROLADOR PD')
sl_tref  = add_slider('T referencia', 'temperatura objetivo de la CPU (°C)',
                       0.895, 40, 80, 60, 1, '#2a78d6')
sl_kp    = add_slider('Kp', 'ganancia proporcional del controlador',
                       0.848, 0.0, 8.0, 2.2, 0.1, '#2a78d6')
sl_kd    = add_slider('Kd', 'ganancia derivativa — frena cambios bruscos',
                       0.801, 0.0, 20, 5.0, 0.5, '#2a78d6')
sl_pwm0  = add_slider('PWM base', 'velocidad de ventilador en régimen (% base)',
                       0.754, 20, 80, 50, 1, '#2a78d6')

# --- Sistema térmico ---
section_label(0.715, 'SISTEMA TÉRMICO')
sl_tamb  = add_slider('T ambiente base', 'temperatura del cuarto (°C)',
                       0.665, 15, 40, 25, 1, '#1baf7a')
sl_gcal  = add_slider('Ganancia calor', 'cuánto calienta el CPU por % de carga',
                       0.618, 0.05, 0.5, 0.22, 0.01, '#e34948')
sl_gcool = add_slider('Ganancia cooler', 'eficiencia base del cooler por % PWM',
                       0.571, 0.05, 0.4, 0.17, 0.01, '#1baf7a')
sl_tau   = add_slider('Inercia térmica', 'qué tan lento responde la temperatura (τ)',
                       0.524, 0.05, 0.5, 0.15, 0.01, '#1baf7a')

# --- Aleatoriedad ---
section_label(0.484, 'COMPORTAMIENTO ALEATORIO')
sl_cpu_var     = add_slider('Variabilidad CPU', 'qué tan seguido cambia la carga del CPU',
                             0.434, 0.0, 1.0, 0.5, 0.05, '#eda100')
sl_perturb     = add_slider('Frec. perturbaciones', 'probabilidad de que aparezca una perturbación',
                             0.387, 0.0, 1.0, 0.4, 0.05, '#eda100')
sl_perturb_int = add_slider('Intensidad perturb.', 'cuán severas son las perturbaciones',
                             0.340, 0.0, 1.0, 0.5, 0.05, '#eda100')

# --- Botones ---
ax_start = fig.add_axes([0.015, 0.285, 0.12, 0.030])
ax_stop  = fig.add_axes([0.155, 0.285, 0.12, 0.030])
ax_reset = fig.add_axes([0.015, 0.245, 0.26, 0.030])

btn_start = Button(ax_start, 'Iniciar',   color='#2a78d6', hovercolor='#1a5cb0')
btn_stop  = Button(ax_stop,  'Pausar',    color='#eda100', hovercolor='#c98500')
btn_reset = Button(ax_reset, 'Reiniciar', color='#555',    hovercolor='#333')

for btn in [btn_start, btn_stop, btn_reset]:
    btn.label.set_color('white')
    btn.label.set_fontsize(9)
    btn.label.set_fontweight('bold')

# ==========================================
# LEYENDA DE PERTURBACIONES
# ==========================================

fig.text(0.015, 0.220, 'PERTURBACIONES', fontsize=7, color='#888',
         style='italic', fontweight='bold')
for i, (lbl, col) in enumerate(zip(PERTURB_LABELS, PERTURB_COLORS)):
    fig.text(0.018, 0.201 - i*0.022, '■', fontsize=10, color=col, va='top')
    fig.text(0.038, 0.203 - i*0.022, lbl, fontsize=7.5, color='#444', va='top')

# indicador de estado
status_text  = fig.text(0.015, 0.085, '[ ] Detenido', fontsize=9,
                         color='#aaa', fontweight='bold', va='top')
cpu_text     = fig.text(0.015, 0.060, 'CPU: – %',          fontsize=8, color='#444', va='top')
perturb_text = fig.text(0.015, 0.040, 'Perturbación: Normal', fontsize=8, color='#444', va='top')
temp_text    = fig.text(0.015, 0.020, 'T CPU: – °C',       fontsize=8, color='#444', va='top')

# ==========================================
# LÓGICA DE SIMULACIÓN
# ==========================================

def next_cpu_target():
    var = sl_cpu_var.val
    lo = max(5,  20 - var*15)
    hi = min(100, 35 + var*65)
    return float(np.random.uniform(lo, hi))

def maybe_trigger_perturb():
    freq  = sl_perturb.val
    inten = sl_perturb_int.val
    if state['perturb_cd'] > 0:
        state['perturb_cd'] -= 1
        return
    if np.random.random() < freq * 0.02:
        p   = np.random.choice([1, 2, 3, 4], p=[0.35, 0.25, 0.20, 0.20])
        dur = int(np.random.uniform(20, 60 + inten*80))
        state['perturb']       = p
        state['perturb_timer'] = dur
        state['perturb_cd']    = dur + int(np.random.uniform(30, 80))
    else:
        state['perturb'] = 0

def step():
    T_ref   = sl_tref.val
    Kp      = sl_kp.val
    Kd      = sl_kd.val
    T_amb_b = sl_tamb.val
    G_CAL   = sl_gcal.val
    G_COOL  = sl_gcool.val
    tau     = sl_tau.val
    inten   = sl_perturb_int.val

    t   = state['t']
    T   = state['T']

# Detección de falla térmica
    if T > 95 or T < 20:
        state['running'] = False
        state['fault'] = True
        state['fault_msg'] = 'Falla térmica  —  T > 95 °C' if T > 95 else 'Falla térmica  —  T < 20 °C'

    # CPU aleatorio — suavizado
    var = sl_cpu_var.val
    if t % max(1, int(30 - var*25)) == 0:
        state['cpu_target'] = next_cpu_target()
    state['cpu_actual'] += (state['cpu_target'] - state['cpu_actual']) * 0.08
    cpu = state['cpu_actual']

    # perturbaciones
    maybe_trigger_perturb()
    perturb = state['perturb']
    if state['perturb_timer'] > 0:
        state['perturb_timer'] -= 1
    else:
        state['perturb'] = 0

    T_amb      = T_amb_b
    eficiencia = 1.0
    extra      = 0.0

    if perturb == 1:   
        T_amb = T_amb_b + 10 + inten*17
    elif perturb == 2: 
        T_amb = T_amb_b + 8 + inten*17
    elif perturb == 3: 
        eficiencia = max(0.3, 0.85 - inten*0.55)
    elif perturb == 4:
        T = T + (1 + inten*3)

    # ── Controlador PD ──────────────────────────────────────────
    error    = T_ref - T
    derivada = T - state['T_ant'] 


    PWM = state['PWM'] + Kp * (-error) + Kd * derivada
    PWM = float(np.clip(PWM, 20, 100))

    # ── Modelo térmico con inercia (τ) ────────────────────────────────────
    calor    = G_CAL  * cpu
    disipado = G_COOL * PWM * eficiencia
    amb_inf  = 0.02 * (T_amb - T)
    delta    = calor - disipado + amb_inf + extra
    
    T_ant = T
    T    += tau * delta

    
    state['t']           = t + 1
    state['T']           = T
    state['PWM']         = PWM
    state['T_ant']       = T_ant

    state['tArr'].append(t)
    state['refArr'].append(round(T_ref, 1))
    state['tempArr'].append(round(T, 2))
    state['errArr'].append(round(abs(error), 2))
    state['pertArr'].append(perturb)
    state['pwmArr'].append(round(cpu, 1))

    # ── Cortamos los valores para no sobrecargar la memoria ────────────────────────────────────

    state['tArr'] = state['tArr'][-TIEMPO_VENTANA:]
    state['refArr'] = state['refArr'][-TIEMPO_VENTANA:]
    state['tempArr'] = state['tempArr'][-TIEMPO_VENTANA:]
    state['errArr'] = state['errArr'][-TIEMPO_VENTANA:]
    state['pertArr'] = state['pertArr'][-TIEMPO_VENTANA:]
    state['pwmArr'] = state['pwmArr'][-TIEMPO_VENTANA:]
    

# ==========================================
# UPDATE GRÁFICOS
# ==========================================

def redraw(_=None):
    tA   = state['tArr']
    tRef = state['refArr']
    tObs = state['tempArr']
    eArr = state['errArr']
    pArr = state['pertArr']
    cArr = state['pwmArr']

    if not tA:
        return

    t_now = tA[-1]
    t_lo  = max(0, t_now - TIEMPO_VENTANA)
    t_hi  = t_now + 5

    idx0 = max(0, len(tA) - TIEMPO_VENTANA)
    tSl  = tA[idx0:]
    rSl  = tRef[idx0:]
    oSl  = tObs[idx0:]
    eSl  = eArr[idx0:]
    pSl  = pArr[idx0:]
    cSl  = cArr[idx0:]

    T_ref = sl_tref.val

    # ax0 θi
    line_ref.set_data(tSl, rSl)
    ax0.set_xlim(t_lo, t_hi)
    ax0.set_ylim(T_ref - 6, T_ref + 6)

    # ax1 θo
    line_temp.set_data(tSl, oSl)
    line_ref2.set_data(tSl, rSl)
    ax1.set_xlim(t_lo, t_hi)
    if oSl:
        mn, mx = min(oSl), max(oSl)
        pad = max(5, (mx - mn)*0.2)
        ax1.set_ylim(mn - pad, mx + pad)

    # ax2 error
    line_err.set_data(tSl, eSl)
    ax2.set_xlim(t_lo, t_hi)
    if eSl:
        em = max(5, max(abs(e) for e in eSl))
        ax2.set_ylim(-em - 2, em + 2)
    for coll in ax2.collections[:]:
        coll.remove()
    if eSl:
        ax2.fill_between(tSl, eSl, 0,
                         where=[e > 0 for e in eSl],
                         alpha=0.1, color='#4a3aa7')
        ax2.fill_between(tSl, eSl, 0,
                         where=[e < 0 for e in eSl],
                         alpha=0.1, color='#e34948')

    # ax3 CPU + perturbaciones
    line_cpu.set_data(tSl, cSl)
    if bar_holder['bars']:
        bar_holder['bars'].remove()
        bar_holder['bars'] = None
    if pSl:
        colors = [PERTURB_COLORS[p] for p in pSl]
        bars = ax3.bar(tSl, [100 if p > 0 else 0 for p in pSl],
                       color=colors, width=1.1, align='edge',
                       alpha=0.25, zorder=2)
        bar_holder['bars'] = bars
    ax3.set_xlim(t_lo, t_hi)
    ax3.set_ylim(0, 105)
    ax3.set_ylabel('CPU %', fontsize=8)

    # anotación perturbación
    p_now = pArr[-1] if pArr else 0
    ann_perturb.set_text(f'  {PERTURB_LABELS[p_now]}  ')
    ann_perturb.get_bbox_patch().set_facecolor(PERTURB_COLORS[p_now] + '33')
    ann_perturb.get_bbox_patch().set_edgecolor(PERTURB_COLORS[p_now])

    # textos estado
    T_cur   = state['T']
    cpu_cur = state['cpu_actual']
    cpu_text.set_text(f'CPU: {cpu_cur:.0f} %')
    perturb_text.set_text(f'Perturbación: {PERTURB_LABELS[p_now]}')
    perturb_text.set_color(PERTURB_COLORS[p_now])
    temp_text.set_text(f'T CPU: {T_cur:.1f} °C')
    temp_color = '#e34948' if T_cur > sl_tref.val + 10 else \
                 '#eda100' if T_cur > sl_tref.val + 3  else '#1baf7a'
    temp_text.set_color(temp_color)

    if state.get('fault'):
        ax1.set_title(state['fault_msg'], fontsize=11, color='#e34948',
                    fontweight='bold', pad=5)
        status_text.set_text('[!] FALLA TÉRMICA')
        status_text.set_color('#e34948')
    
    fig.canvas.draw_idle()




def on_timer():
    if not state['running']:
        return
    for _ in range(2):
        step()
    redraw()

# ==========================================
# BOTONES
# ==========================================

def start(event):
    if not state['running']:
        state['running'] = True
        status_text.set_text('[>] Corriendo')
        status_text.set_color('#1baf7a')
        fig.canvas.draw_idle()

def stop(event):
    state['running'] = False
    status_text.set_text('[||] Pausado')
    status_text.set_color('#eda100')
    fig.canvas.draw_idle()

def reset(event):
    state['running']       = False
    state['t']             = 0
    state['T']             = sl_tref.val - 20
    state['T_ant']         = sl_tref.val - 20
    state['PWM']           = sl_pwm0.val
    state['tArr'].clear()
    state['tempArr'].clear()
    state['refArr'].clear()
    state['errArr'].clear()
    state['pertArr'].clear()
    state['pwmArr'].clear()
    state['cpu_actual']    = 20.0
    state['cpu_target']    = 20.0
    state['perturb']       = 0
    state['perturb_timer'] = 0
    state['perturb_cd']    = 0
    state['fault'] = False
    state['fault_msg'] = ''
    ax1.set_title('θo(t) — Temperatura real CPU', fontsize=9, color='#444', pad=3)
    
    for ax in CHART_AXES:
        for coll in ax.collections:
            coll.remove()
    bar_holder['bars'] = None
    for line in [line_ref, line_temp, line_ref2, line_err, line_cpu]:
        line.set_data([], [])
    status_text.set_text('[ ] Detenido')
    status_text.set_color('#aaa')
    fig.canvas.draw_idle()

btn_start.on_clicked(start)
btn_stop.on_clicked(stop)
btn_reset.on_clicked(reset)

# ==========================================
# TIMER
# ==========================================

timer = fig.canvas.new_timer(interval=INTERVAL_MS)
timer.add_callback(on_timer)
timer.start()

plt.show()