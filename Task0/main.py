from math import cos, pi, prod, isclose
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk

def jordan_gauss_solve(matrix: list[list[float]], eps: float = 1e-12) -> list[float]:
    A = deepcopy(matrix)
    n = len(A)
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[max_row][i]) < eps:
            raise ValueError("Система вырождена или коэффициент слишком мал для устойчивого решения.")
        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
        pivot = A[i][i]
        A[i] = [val / pivot for val in A[i]]
        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            if factor != 0:
                A[r] = [A[r][c] - factor * A[i][c] for c in range(n + 1)]
    # Посл столец
    return [A[i][n] for i in range(n)]

def uniform_grid(a: float, b: float, n: int) -> list[float]:
    return [a + (b - a) * i / (n - 1) for i in range(n)] if n > 1 else [a]

def chebyshev_grid(a: float, b: float, n: int) -> list[float]:
    return [(a + b) / 2 + (b - a) / 2 * cos((2 * i + 1) * pi / (2 * n)) for i in range(n)]

class FunctionDef:
    def __init__(self, a: float, b: float, name: str, func_callable):
        self.a = a
        self.b = b
        self.name = name
        self._f = func_callable

    def evaluate(self, x: float, n: int = 0) -> float:
        return float(self._f(x, n))

# Методы интерполяции
def solve_via_matrix(function: FunctionDef, n: int, grid: list[float]):
    A = [[0.0 for _ in range(n + 1)] for _ in range(n)]
    y = [0.0 for _ in range(n)]
    for i in range(n):
        xi = grid[i]
        y[i] = function.evaluate(xi, n)
        for j in range(n):
            A[i][j] = xi ** (n - j - 1)
    for i in range(n):
        A[i][n] = y[i]
    coeffs = jordan_gauss_solve(A)
    # polinom: sum coeffs[i] * x^(n-i-1)
    return lambda x: sum(coeffs[i] * x ** (n - i - 1) for i in range(n)), coeffs

def solve_via_lagrange(function: FunctionDef, n: int, grid: list[float]):
    y = [function.evaluate(grid[i], n) for i in range(n)]

    def phi(i, x):
        terms = []
        xi = grid[i]
        for j in range(n):
            if j == i:
                continue
            denom = (xi - grid[j])
            if denom == 0:
                # Сетка некорректна (повторные узлы)
                raise ValueError("Повторяющиеся узлы в сетке при вычислении Лагранжа.")
            terms.append((x - grid[j]) / denom)
        return prod(terms)

    def lagrange_pol(x):
        return sum(y[i] * phi(i, x) for i in range(n))
    return lagrange_pol, y

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tipwindow:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#fff8dc", relief=tk.SOLID, borderwidth=1,
                       font=("Segoe UI", 9))
        lbl.pack(ipadx=6, ipady=3)

    def hide(self, _=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


# Мейн часть
class InterpolationApp:
    def __init__(self, root: ThemedTk):
        self.root = root
        self.root.title("Визуализация")
        self.root.geometry("1100x760")
        self.root.minsize(920, 640)
        self.style = ttk.Style(self.root)
        self.style.theme_use("equilux")
        self.style.configure("Header.TLabel", font=("Segoe UI Semibold", 18))
        self.style.configure("Sub.TLabel", font=("Segoe UI", 11))
        self.style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        self.style.configure("TFrame", background="#2a2a2a")
        self.style.configure("TLabel", background="#2a2a2a", foreground="#ffffff")
        self.root.configure(bg="#2a2a2a")

        # Дефолтные функции
        self.functions = [
            FunctionDef(-1, 1, "Полином x^{n-1} + 3x^{n-2} - 1", lambda x, n: x ** (n - 1) + 3 * x ** (n - 2) - 1),
            FunctionDef(-5, 5, "Рациональная 1/(25x^2+1)", lambda x, n: 1 / (25 * x ** 2 + 1) if x != 0 else 1),
            FunctionDef(-1, 2, "Модуль |x|", lambda x, n: abs(x))
        ]
        self.current_function = self.functions[0]
        self.grid_type = tk.StringVar(value="uniform")
        self.method = tk.StringVar(value="matrix")
        self.node_count = tk.IntVar(value=10)

        self._build_ui()
        self._init_plot()

    def _build_ui(self):
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=14, pady=(12, 8))

        ttk.Label(header, text="Визуализация интерполяции", style="Sub.TLabel").pack(side=tk.LEFT, padx=12)

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)
        left = ttk.Frame(main, width=360)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), pady=6)
        left.pack_propagate(False)
        block_func = ttk.LabelFrame(left, text="1. Функция", padding=12)
        block_func.pack(fill=tk.X, pady=8)
        for f in self.functions:
            r = ttk.Radiobutton(block_func, text=f.name, value=f.name,
                                command=lambda ff=f: self._select_function(ff),
                                variable=tk.StringVar(value=""))
            # используем стиль, но ставим .pack вручную
            r.pack(anchor=tk.W, pady=3)
            ToolTip(r, f"Диапазон: [{f.a}, {f.b}]")
        # ТЕкущая формула
        self.func_label = ttk.Label(block_func, text=f"Текущее: {self.current_function.name}", style="Sub.TLabel")
        self.func_label.pack(anchor=tk.W, pady=(8, 0))

        # Блок сетки и узлов
        block_grid = ttk.LabelFrame(left, text="2. Сетка и узлы", padding=12)
        block_grid.pack(fill=tk.X, pady=8)

        # количеств узлов
        nodes_frame = ttk.Frame(block_grid)
        nodes_frame.pack(fill=tk.X, pady=6)
        ttk.Label(nodes_frame, text="Количество узлов:").pack(side=tk.LEFT)
        self.nodes_spin = ttk.Spinbox(nodes_frame, from_=2, to=1000, textvariable=self.node_count, width=8)
        self.nodes_spin.pack(side=tk.LEFT, padx=(8, 6))
        self.nodes_slider = ttk.Scale(nodes_frame, from_=2, to=500, orient=tk.HORIZONTAL,
                                      command=self._on_slider_change)
        self.nodes_slider.set(self.node_count.get())
        self.nodes_slider.pack(fill=tk.X, expand=True, padx=(2, 6))
        ToolTip(self.nodes_spin, "Введите целое число узлов (минимум 2).")

        # сетка
        grid_box = ttk.Frame(block_grid)
        grid_box.pack(fill=tk.X, pady=(8, 0))
        ttk.Radiobutton(grid_box, text="Равномерная", variable=self.grid_type, value="uniform").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(grid_box, text="Чебышевская", variable=self.grid_type, value="chebyshev").pack(side=tk.LEFT, padx=6)
        ToolTip(grid_box, "Выберите вид расположения узлов.")

        # методы
        block_method = ttk.LabelFrame(left, text="3. Метод", padding=12)
        block_method.pack(fill=tk.X, pady=8)
        ttk.Radiobutton(block_method, text="Матричный (стандартный)", variable=self.method, value="matrix").pack(anchor=tk.W, pady=4)
        ttk.Radiobutton(block_method, text="Лагранж (формула)", variable=self.method, value="lagrange").pack(anchor=tk.W, pady=4)
        ToolTip(block_method, "Матричный метод решает систему coef*x = y; Лагранж использует базис полиномов L_i(x).")

        # батоны
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=(8, 12))
        self.btn_plot = ttk.Button(btn_frame, text="Построить график", style="Accent.TButton", command=self.build_and_plot)
        self.btn_plot.pack(fill=tk.X, pady=(0, 6))
        self.btn_clear = ttk.Button(btn_frame, text="Сбросить вид графика", command=self._reset_plot)
        self.btn_clear.pack(fill=tk.X)

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Область графика
        plot_frame = ttk.LabelFrame(right, text="График", padding=8)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 0), pady=(0, 8))

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Логи/резы
        info_frame = ttk.LabelFrame(right, text="Результаты & Диагностика", padding=8)
        info_frame.pack(fill=tk.X, expand=False)
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, background="#1e1e1e", foreground="#eaeaea",
                                 font=("Consolas", 10))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.insert(tk.END, "Готово. Выберите параметры и нажмите «Построить график».\n")
        self.info_text.config(state=tk.DISABLED)

    def _init_plot(self):
        # начальная сетка
        self.ax.clear()
        self.ax.set_title("График функции и приближения")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()

    def _select_function(self, f: FunctionDef):
        self.current_function = f
        self.func_label.config(text=f"Текущее: {f.name}  [{f.a}, {f.b}]")
        self.log(f"Выбрана функция: {f.name} с диапазоном [{f.a}, {f.b}]")

    def _on_slider_change(self, val):
        # синхронизируем slider и spinbox
        try:
            n = int(float(val))
        except ValueError:
            return
        self.node_count.set(n)
        self.nodes_spin.delete(0, tk.END)
        self.nodes_spin.insert(0, str(n))

    def _reset_plot(self):
        self._init_plot()
        self.log("График очищен.")

    def log(self, text: str):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, f"{text}\n")
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)

    def build_and_plot(self):
        # валидация
        try:
            N = int(self.node_count.get())
        except Exception:
            messagebox.showerror("Ошибка ввода", "Количество узлов должно быть целым числом.")
            return
        if N < 2:
            messagebox.showerror("Ошибка", "Количество узлов должно быть не менее 2.")
            return
        # определить сетку
        grid_choice = self.grid_type.get()
        if grid_choice == "uniform":
            X = uniform_grid(self.current_function.a, self.current_function.b, N)
        else:
            X = chebyshev_grid(self.current_function.a, self.current_function.b, N)

        method_choice = self.method.get()
        try:
            if method_choice == "matrix":
                approx_func, coeffs = solve_via_matrix(self.current_function, N, X)
                self.log(f"Матричный метод: найдено {len(coeffs)} коэффициентов.")
                # показать несколько коэффициентов
                self.log("Коэффициенты (старшие -> младшие): " + ", ".join(f"{c:.4g}" for c in coeffs[:min(8, len(coeffs))]) + ("" if len(coeffs)<=8 else ", ..."))
            else:
                approx_func, yvals = solve_via_lagrange(self.current_function, N, X)
                self.log(f"Лагранж: использовано {N} узлов.")
        except Exception as e:
            messagebox.showerror("Ошибка численного метода", str(e))
            self.log("Ошибка при вычислениях: " + str(e))
            return


        a, b = self.current_function.a, self.current_function.b
        XX = np.linspace(a, b, 1000)
        YY = np.array([self.current_function.evaluate(x, N) for x in XX])
        YY_app = np.array([approx_func(x) for x in XX])


        max_dev_on_grid = max(abs(self.current_function.evaluate(xi, N) - approx_func(xi)) for xi in X)
        max_dev_on_segment = float(np.max(np.abs(YY - YY_app)))

        #  Отрисовка
        self.ax.clear()
        self.ax.plot(XX, YY, label=f"{self.current_function.name} (исходная)", linewidth=1.5)
        self.ax.plot(XX, YY_app, label="Приближение", linewidth=1.2)
        # показать узлы
        node_y = [self.current_function.evaluate(xi, N) for xi in X]
        self.ax.scatter(X, node_y, marker='o', s=30, zorder=5, label="Узлы")
        self.ax.set_title(f"{self.current_function.name} — {method_choice.capitalize()} — N={N}")
        self.ax.grid(True, linestyle='--', alpha=0.4)
        self.ax.legend()
        self.canvas.draw()

        # Логи результатов
        self.log(f"Использована сетка: {('равномерная' if grid_choice=='uniform' else 'чебышевская')}. N = {N}")
        self.log(f"Максимальное отклонение на узлах: {max_dev_on_grid:.3e}")
        self.log(f"Максимальное отклонение на отрезке [{a},{b}]: {max_dev_on_segment:.3e}")

        # Предупреждение о больших отклонениях
        if not isclose(max_dev_on_grid, 0.0, rel_tol=1e-9, abs_tol=1e-9) and max_dev_on_grid > 1e-6:
            self.log("Внимание: отклонение на узлах существенно больше машинной погрешности — проверьте устойчивость метода и число узлов.")


def main():
    root = ThemedTk(theme="equilux")
    app = InterpolationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()















