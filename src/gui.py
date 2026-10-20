import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkfont

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.models import Perceptron, Adaline, evaluate_predictions
from src.visualization import plot_decision_boundary, plot_convergence
from src.preparingData import data, get_train_test_split

features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
classPairs = ['Adelie vs Chinstrap', 'Adelie vs Gentoo', 'Chinstrap vs Gentoo']
seed = 42

INK = '#0a2030'
OCEAN = '#0f2f45'
OCEAN_LIGHT = '#173f58'
OCEAN_LINE = '#285670'
ICE = '#e8f1f6'
MIST = '#93afc1'
DIM = '#5f7f93'

CYAN = '#52c6e3'
CYAN_HOVER = '#7bd6ec'
CYAN_PRESS = '#3aaecb'

GLACIER = '#eaf1f6'
SNOW = '#ffffff'
LINE = '#d3dfe8'
LINE_STRONG = '#b4c6d3'
SLATE = '#2f4a5f'
SLATE_MUTED = '#5a7385'

LOG_INFO = '#9db6c6'
LOG_TRAIN = CYAN
LOG_TEST = '#6fe0a3'
LOG_CLASSIFY = '#c9aaff'
LOG_RULE = '#2b536b'

PILL_IDLE = ('#dde8ef', '#4a6577')
PILL_READY = ('#d4f2e3', '#12603b')


def _pick_font(root, candidates, fallback):
    installed = set(tkfont.families(root))
    for name in candidates:
        if name in installed:
            return name
    return fallback


class PenguinGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Penguin classifier')
        self.configure(bg=GLACIER)

        height = min(950, self.winfo_screenheight() - 80)
        self.geometry(f'1380x{height}')
        self.minsize(1200, min(800, height))

        self.model = None
        self.split = None
        self.class_names = None
        self.last_params = None

        self._setup_fonts()
        self._setup_style()
        self._build_sidebar()
        self._build_main()
        self._show_placeholder()
        self.log('System ready. Choose features and a class pair, then press Train model.', 'info')

    def _setup_fonts(self):
        ui = _pick_font(self, ['Segoe UI', 'SF Pro Text', 'Helvetica Neue', 'Inter', 'Ubuntu', 'Noto Sans'], 'Helvetica')
        mono = _pick_font(self, ['Cascadia Mono', 'Consolas', 'SF Mono', 'Menlo', 'DejaVu Sans Mono'], 'Courier')

        self.f_body = (ui, 10)
        self.f_small = (ui, 9)
        self.f_bold = (ui, 10, 'bold')
        self.f_head = (ui, 11, 'bold')
        self.f_title = (ui, 16, 'bold')
        self.f_run = (ui, 14, 'bold')
        self.f_metric = (ui, 13, 'bold')
        self.f_mono = (mono, 10)
        self.f_mono_bold = (mono, 10, 'bold')

    def _setup_style(self):
        s = ttk.Style(self)
        if 'clam' in s.theme_names():
            s.theme_use('clam')
        s.configure('.', font=self.f_body)

        field = dict(fieldbackground=OCEAN_LIGHT, foreground=ICE, bordercolor=OCEAN_LINE,lightcolor=OCEAN_LINE, darkcolor=OCEAN_LINE, borderwidth=1, padding=(8, 6))
        focus = dict(bordercolor=[('focus', CYAN)], lightcolor=[('focus', CYAN)], darkcolor=[('focus', CYAN)])

        s.configure('Side.TEntry', insertcolor=ICE, **field)
        s.map('Side.TEntry', fieldbackground=[('disabled', OCEAN)], foreground=[('disabled', DIM)], **focus)

        s.configure('Side.TCombobox', arrowcolor=ICE, background=OCEAN_LIGHT,
                    selectbackground=OCEAN_LIGHT, selectforeground=ICE, **field)
        s.map('Side.TCombobox',
              fieldbackground=[('readonly', OCEAN_LIGHT)],
              foreground=[('readonly', ICE)],
              background=[('active', OCEAN_LINE), ('pressed', OCEAN_LINE)],
              selectbackground=[('readonly', OCEAN_LIGHT)],
              selectforeground=[('readonly', ICE)], **focus)

        self.option_add('*TCombobox*Listbox.background', OCEAN_LIGHT)
        self.option_add('*TCombobox*Listbox.foreground', ICE)
        self.option_add('*TCombobox*Listbox.selectBackground', CYAN)
        self.option_add('*TCombobox*Listbox.selectForeground', INK)
        self.option_add('*TCombobox*Listbox.font', self.f_body)
        self.option_add('*TCombobox*Listbox.borderWidth', 0)

        def flat(name, bg, fg, hover, press, font, pad):
            s.configure(name, background=bg, foreground=fg, bordercolor=bg, lightcolor=bg, darkcolor=bg,
                        borderwidth=0, focusthickness=0, focuscolor=bg, padding=pad, font=font, relief='flat')
            s.map(name,background=[('pressed', press), ('active', hover)],bordercolor=[('pressed', press), ('active', hover)], lightcolor=[('pressed', press), ('active', hover)], darkcolor=[('pressed', press), ('active', hover)])

        flat('Primary.TButton', CYAN, INK, CYAN_HOVER, CYAN_PRESS, self.f_bold, (14, 10))
        flat('Ghost.TButton', OCEAN_LIGHT, ICE, '#1f5273', OCEAN_LINE, self.f_body, (12, 8))
        flat('Link.TButton', INK, MIST, INK, INK, self.f_small, (8, 2))
        s.map('Link.TButton', foreground=[('active', ICE)])

        s.configure('Seg.Toolbutton', background=OCEAN_LIGHT, foreground=MIST, bordercolor=OCEAN_LIGHT,lightcolor=OCEAN_LIGHT, darkcolor=OCEAN_LIGHT, borderwidth=0, focusthickness=0,focuscolor=OCEAN_LIGHT, padding=(10, 8), anchor='center', font=self.f_bold, relief='flat')
        s.map('Seg.Toolbutton',background=[('selected', CYAN), ('active', '#1f5273')],foreground=[('selected', INK), ('active', ICE)],bordercolor=[('selected', CYAN)], lightcolor=[('selected', CYAN)], darkcolor=[('selected', CYAN)],relief=[('selected', 'flat'), ('pressed', 'flat')])
        s.configure('Side.TCheckbutton', background=OCEAN, foreground=ICE, focuscolor=OCEAN,indicatorbackground=OCEAN_LIGHT, indicatorforeground=INK,upperbordercolor=OCEAN_LINE, lowerbordercolor=OCEAN_LINE, padding=(2, 2))
        s.map('Side.TCheckbutton', background=[('active', OCEAN)], indicatorbackground=[('selected', CYAN), ('active', OCEAN_LINE)], indicatorforeground=[('selected', INK)])

        s.configure('Console.Vertical.TScrollbar', background=OCEAN_LINE, troughcolor=INK, bordercolor=INK,lightcolor=OCEAN_LINE, darkcolor=OCEAN_LINE, arrowcolor=MIST, relief='flat', gripcount=0)
        s.map('Console.Vertical.TScrollbar', background=[('active', '#3a6c87')])

    def _section(self, parent, title):
        tk.Frame(parent, bg=OCEAN_LINE, height=1).pack(fill='x', pady=(12, 8))
        tk.Label(parent, text=title, bg=OCEAN, fg=ICE, font=self.f_head, anchor='w').pack(fill='x', pady=(0, 6))
        grid = tk.Frame(parent, bg=OCEAN)
        grid.pack(fill='x')
        grid.columnconfigure((0, 1), weight=1, uniform='side')
        return grid

    def _labeled(self, grid, text, row, col, span=1):
        padx = (0, 0) if span == 2 else ((0, 6) if col == 0 else (6, 0))
        cell = tk.Frame(grid, bg=OCEAN)
        cell.grid(row=row, column=col, columnspan=span, sticky='ew', padx=padx, pady=(0, 8))
        caption = tk.Label(cell, text=text, bg=OCEAN, fg=MIST, font=self.f_small, anchor='w')
        caption.pack(fill='x', pady=(0, 3))
        return cell, caption

    def _combo(self, parent, values, default):
        cmb = ttk.Combobox(parent, values=values, state='readonly', style='Side.TCombobox', width=10)
        cmb.set(default)
        cmb.pack(fill='x')
        cmb.bind('<<ComboboxSelected>>', lambda e: e.widget.selection_clear())
        return cmb

    def _entry(self, parent, default=''):
        ent = ttk.Entry(parent, style='Side.TEntry', width=10)
        if default:
            ent.insert(0, default)
        ent.pack(fill='x')
        return ent

    def _build_sidebar(self):
        side = tk.Frame(self, bg=OCEAN, width=336)
        side.pack(side='left', fill='y')
        side.pack_propagate(False)
        body = tk.Frame(side, bg=OCEAN)
        body.pack(fill='both', expand=True, padx=22, pady=12)
        tk.Label(body, text='Penguin classifier', bg=OCEAN, fg=ICE, font=self.f_title, anchor='w').pack(fill='x', pady=(0, 5))

        grid = self._section(body, 'Data configuration')
        cell, _ = self._labeled(grid, 'Feature 1', 0, 0)
        self.cmb_f1 = self._combo(cell, features, 'CulmenLength')
        cell, _ = self._labeled(grid, 'Feature 2', 0, 1)
        self.cmb_f2 = self._combo(cell, features, 'CulmenDepth')
        cell, _ = self._labeled(grid, 'Class pair', 1, 0, span=2)
        self.cmb_classes = self._combo(cell, classPairs, 'Adelie vs Gentoo')

        grid = self._section(body, 'Model configuration')
        self.var_algo = tk.StringVar(value='Perceptron')
        seg = tk.Frame(grid, bg=OCEAN)
        seg.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        seg.columnconfigure((0, 1), weight=1, uniform='seg')

        ttk.Radiobutton(seg, text='Perceptron', value='Perceptron', variable=self.var_algo, style='Seg.Toolbutton', command=self._on_algo_change).grid(row=0, column=0, sticky='ew')
        ttk.Radiobutton(seg, text='Adaline', value='Adaline', variable=self.var_algo, style='Seg.Toolbutton', command=self._on_algo_change).grid(row=0, column=1, sticky='ew')

        cell, _ = self._labeled(grid, 'Learning rate (eta)', 1, 0)
        self.ent_eta = self._entry(cell, '0.01')
        cell, _ = self._labeled(grid, 'Epochs', 1, 1)
        self.ent_epochs = self._entry(cell, '1000')
        cell, _ = self._labeled(grid, 'Mse threshold', 2, 0)
        self.ent_mse = self._entry(cell, '0.01')

        self.var_bias = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid, text='Include bias', variable=self.var_bias, style='Side.TCheckbutton').grid( row=2, column=1, sticky='sw', padx=(6, 0), pady=(0, 12))
        self._on_algo_change()

        grid = self._section(body, 'Execution')
        ttk.Button(grid, text='Train model', style='Primary.TButton', command=self.on_train).grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Button(grid, text='Test model', style='Ghost.TButton', command=self.on_test).grid( row=1, column=0, sticky='ew', padx=(0, 4))
        ttk.Button(grid, text='Save plot', style='Ghost.TButton', command=self.on_save).grid(row=1, column=1, sticky='ew', padx=(4, 0))
        grid = self._section(body, 'Classify one sample')
        cell, self.lbl_s1 = self._labeled(grid, 'Feature 1 value', 0, 0)
        self.ent_s1 = self._entry(cell)
        cell, self.lbl_s2 = self._labeled(grid, 'Feature 2 value', 0, 1)
        self.ent_s2 = self._entry(cell)
        ttk.Button(grid, text='Classify sample', style='Primary.TButton', command=self.on_classify).grid(row=1, column=0, columnspan=2, sticky='ew', pady=(10, 0), ipady=4)

    def _on_algo_change(self):
        if self.var_algo.get() == 'Adaline':
            self.ent_mse.state(['!disabled'])
        else:
            self.ent_mse.state(['disabled'])

    @staticmethod
    def _panel(parent):
        outer = tk.Frame(parent, bg=LINE)
        inner = tk.Frame(outer, bg=SNOW)
        inner.pack(fill='both', expand=True, padx=1, pady=1)
        return outer, inner

    def _build_main(self):
        main = tk.Frame(self, bg=GLACIER)
        main.pack(side='left', fill='both', expand=True, padx=24, pady=(18, 20))

        header = tk.Frame(main, bg=GLACIER)
        header.pack(fill='x')
        titles = tk.Frame(header, bg=GLACIER)
        titles.pack(side='left', fill='x', expand=True)
        self.var_run_title = tk.StringVar(value='No model trained yet')
        self.var_run_sub = tk.StringVar(value='Your results will appear here after you train a model.')
        tk.Label(titles, textvariable=self.var_run_title, bg=GLACIER, fg=INK, font=self.f_run, anchor='w').pack(fill='x')
        tk.Label(titles, textvariable=self.var_run_sub, bg=GLACIER, fg=SLATE_MUTED, font=self.f_body, anchor='w').pack(fill='x')
        self.pill = tk.Label(header, text='Not trained', bg=PILL_IDLE[0], fg=PILL_IDLE[1], font=self.f_bold, padx=14, pady=5)
        self.pill.pack(side='right', anchor='n')

        outer, strip = self._panel(main)
        outer.pack(fill='x', pady=(14, 12))
        self.var_m_algo = tk.StringVar(value='-')
        self.var_m_epochs = tk.StringVar(value='-')
        self.var_m_error = tk.StringVar(value='-')
        self.var_m_acc = tk.StringVar(value='-')

        metrics = (('Algorithm', self.var_m_algo), ('Epochs run', self.var_m_epochs), ('Final error', self.var_m_error), ('Test accuracy', self.var_m_acc))

        for i, (caption, var) in enumerate(metrics):
            if i:
                tk.Frame(strip, bg=LINE, width=1).grid(row=0, column=2 * i - 1, sticky='ns', pady=10)
            cell = tk.Frame(strip, bg=SNOW)
            cell.grid(row=0, column=2 * i, sticky='ew', padx=18, pady=10)
            strip.columnconfigure(2 * i, weight=1, uniform='metric')
            tk.Label(cell, text=caption, bg=SNOW, fg=SLATE_MUTED, font=self.f_small, anchor='w').pack(fill='x')
            tk.Label(cell, textvariable=var, bg=SNOW, fg=INK, font=self.f_metric, anchor='w').pack(fill='x')

        console = tk.Frame(main, bg=INK)
        console.pack(side='bottom', fill='x', pady=(12, 0))
        bar = tk.Frame(console, bg=INK)
        bar.pack(fill='x', padx=14, pady=(8, 0))
        tk.Label(bar, text='Activity log', bg=INK, fg=MIST, font=self.f_bold).pack(side='left')
        ttk.Button(bar, text='Clear', style='Link.TButton', command=self._clear_log).pack(side='right')

        text_row = tk.Frame(console, bg=INK)
        text_row.pack(fill='x', padx=(14, 4), pady=(2, 10))

        self.log_box = tk.Text(text_row, height=5, wrap='none', state='disabled', font=self.f_mono, bg=INK, fg=ICE,  relief='flat', borderwidth=0, highlightthickness=0, padx=0, pady=4,  insertbackground=ICE, selectbackground=OCEAN_LINE, spacing1=1, spacing3=1)
        sb = ttk.Scrollbar(text_row, orient='vertical', command=self.log_box.yview, style='Console.Vertical.TScrollbar')
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.log_box.pack(side='left', fill='both', expand=True)

        self.log_box.tag_config('info', foreground=LOG_INFO)
        self.log_box.tag_config('rule', foreground=LOG_RULE)
        self.log_box.tag_config('train', foreground=LOG_TRAIN, font=self.f_mono_bold)
        self.log_box.tag_config('test', foreground=LOG_TEST, font=self.f_mono_bold)
        self.log_box.tag_config('classify', foreground=LOG_CLASSIFY, font=self.f_mono_bold)
        self.log_box.tag_config('matrix', foreground=ICE, font=self.f_mono)

        outer, plot_panel = self._panel(main)
        outer.pack(side='top', fill='both', expand=True)

        try:
            self.fig = Figure(figsize=(12, 10), dpi=100, layout='constrained')
        except TypeError:
            self.fig = Figure(figsize=(12, 10), dpi=100, constrained_layout=True)

        self.fig.patch.set_facecolor(SNOW)
        try:
            self.fig.get_layout_engine().set(h_pad=0.15, w_pad=0.1)
        except Exception:
            pass

        gs = self.fig.add_gridspec(2, 1, height_ratios=[2.8, 1])
        self.ax_boundary = self.fig.add_subplot(gs[0])
        self.ax_convergence = self.fig.add_subplot(gs[1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas.get_tk_widget().configure(bg=SNOW, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)

    def _show_placeholder(self):
        messages = ((self.ax_boundary, 'Configure parameters and press Train model.'),(self.ax_convergence, 'The convergence curve will display here.'))
        for ax, msg in messages:
            ax.clear()
            ax.set_axis_off()
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
                    color=SLATE_MUTED, fontsize=11, linespacing=1.5)
        self.canvas.draw()

    @staticmethod
    def _polish_axes(ax, open_frame=False):
        for name, spine in ax.spines.items():
            spine.set_color(LINE_STRONG)
            if open_frame and name in ('top', 'right'):
                spine.set_visible(False)
        ax.tick_params(colors=SLATE, labelsize=9)
        ax.xaxis.label.set_color(SLATE)
        ax.yaxis.label.set_color(SLATE)
        ax.title.set_color(INK)

    def _set_status(self, text, ready):
        bg, fg = PILL_READY if ready else PILL_IDLE
        self.pill.configure(text=text, bg=bg, fg=fg)

    def _read_params(self):
        f1, f2 = self.cmb_f1.get(), self.cmb_f2.get()
        if f1 == f2:
            messagebox.showerror('Input error', 'Feature 1 and feature 2 must differ.')
            return None
        c1, c2 = self.cmb_classes.get().split(' vs ')
        try:
            eta = float(self.ent_eta.get())
            epochs = int(self.ent_epochs.get())
            mse_thr = float(self.ent_mse.get())
            if eta <= 0 or epochs <= 0 or mse_thr < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror('Input error', 'Need numeric values > 0 for hyperparameters.')
            return None
        return dict(f1=f1, f2=f2, c1=c1, c2=c2, eta=eta, epochs=epochs, mse_threshold=mse_thr,
                    use_bias=self.var_bias.get(), algorithm=self.var_algo.get())

    @staticmethod
    def _predict_classes(model, X):
        X_arr = np.asarray(X, dtype=float)
        if hasattr(model, 'predict_class'):
            return np.asarray(model.predict_class(X_arr))
        return np.asarray(model.predict(X_arr))

    def log(self, msg, tag='info'):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', msg + '\n', tag)
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _clear_log(self):
        self.log_box.configure(state='normal')
        self.log_box.delete('1.0', 'end')
        self.log_box.configure(state='disabled')

    def on_train(self):
        p = self._read_params()
        if p is None:
            return
        X_tr, y_tr, X_te, y_te = get_train_test_split(data, p['c1'], p['c2'], p['f1'], p['f2'])
        np.random.seed(seed)

        if p['algorithm'] == 'Perceptron':
            model = Perceptron(learning_rate=p['eta'], epochs=p['epochs'], use_bias=p['use_bias'])
        else:
            model = Adaline(learning_rate=p['eta'], epochs=p['epochs'], mse_threshold=p['mse_threshold'], use_bias=p['use_bias'])
        model.fit(X_tr, y_tr)
        self.model, self.split = model, (X_tr, y_tr, X_te, y_te)
        self.class_names, self.last_params = (p['c1'], p['c2']), p
        for ax in (self.ax_boundary, self.ax_convergence):
            ax.clear()
            ax.set_axis_on()

        plot_decision_boundary(X_tr, y_tr, X_te, y_te, model, p['f1'], p['f2'], class1_name=p['c1'],
                               class2_name=p['c2'], title=f"{p['algorithm']}: {p['c1']} vs {p['c2']}", ax=self.ax_boundary)
        plot_convergence(model, p['algorithm'], ax=self.ax_convergence)
        self._polish_axes(self.ax_boundary)
        self._polish_axes(self.ax_convergence, open_frame=True)
        self.canvas.draw()

        ran = len(model.mse_history) if p['algorithm'] == 'Adaline' else len(model.errors)
        if p['algorithm'] == 'Adaline':
            final_error = f"MSE {model.mse_history[-1]:.4f}"
        else:
            final_error = f"{model.errors[-1]} errors"

        bias_text = 'with bias' if p['use_bias'] else 'without bias'
        self.var_run_title.set(f"{p['algorithm']} on {p['c1']} vs {p['c2']}")
        self.var_run_sub.set(f"{p['f1']} and {p['f2']}, {bias_text}, learning rate {p['eta']}")
        self.var_m_algo.set(p['algorithm'])
        self.var_m_epochs.set(f"{ran} of {p['epochs']}")
        self.var_m_error.set(final_error)
        self.var_m_acc.set('Not tested yet')
        self.lbl_s1.configure(text=f"{p['f1']} value")
        self.lbl_s2.configure(text=f"{p['f2']} value")
        self._set_status('Model trained', True)

        self.log('─' * 75, 'rule')
        self.log(f"[Train] {p['algorithm']} ({p['c1']} vs {p['c2']})", 'train')
        self.log(f"Inputs: {p['f1']}, {p['f2']} | eta={p['eta']} | bias={p['use_bias']} | epochs={ran}/{p['epochs']}", 'info')
        self.log(f"Final training state = {final_error}", 'info')

    def on_test(self):
        if self.model is None:
            messagebox.showwarning('Not trained', 'Train a model first.')
            return

        _, _, X_te, y_te = self.split
        c1, c2 = self.class_names
        y_pred = self._predict_classes(self.model, X_te)
        cm, acc = evaluate_predictions(y_te, y_pred)
        total = len(y_te)
        correct = cm['TP'] + cm['TN']

        self.var_m_acc.set(f'{acc * 100:.1f}%')

        self.log('─' * 75, 'rule')
        self.log(f"[Test] Evaluation on {total} test samples", 'test')
        self.log(f"Overall accuracy = {acc * 100:.2f}% ({correct}/{total} correct)\n", 'info')

        col_w = 14
        self.log(" " * col_w + f"Pred: {c1}".ljust(col_w) + f"Pred: {c2}".ljust(col_w), 'matrix')
        self.log(f"True: {c1}".ljust(col_w) + str(cm['TP']).ljust(col_w) + str(cm['FN']).ljust(col_w), 'matrix')
        self.log(f"True: {c2}".ljust(col_w) + str(cm['FP']).ljust(col_w) + str(cm['TN']).ljust(col_w), 'matrix')

    def on_classify(self):
        if self.model is None:
            messagebox.showwarning('Not trained', 'Train a model first.')
            return
        try:
            v1, v2 = float(self.ent_s1.get()), float(self.ent_s2.get())
        except ValueError:
            messagebox.showerror('Input error', 'Enter numeric values for the two selected features.')
            return

        p = self.last_params
        pred = self._predict_classes(self.model, [[v1, v2]])[0]
        name = p['c1'] if pred == 1 else p['c2']

        self.log('─' * 75, 'rule')
        self.log(f"[Classify] Custom input: {p['f1']}={v1}, {p['f2']}={v2}", 'classify')
        self.log(f"Prediction: {name} (Class {int(pred):+d})", 'classify')

    def on_save(self):
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile='penguin_plot.png',
                                            filetypes=[('PNG image', '*.png')])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            self.log('─' * 75, 'rule')
            self.log(f"[Save] Figure saved to: {path}", 'info')

if __name__ == '__main__':
    PenguinGUI().mainloop()