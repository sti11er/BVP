import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import root
import matplotlib.pyplot as plt
import re
import pandas as pd
from PySide6.QtWidgets import QSizePolicy, QHeaderView, QProgressBar, QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QComboBox, QHBoxLayout, QGridLayout, QTextEdit, QDialog, QSpinBox, QFrame, QStyleFactory, QSplitter, QScrollArea, QGroupBox
from PySide6.QtGui import QKeySequence, QPalette, QColor, QFont, QIcon, QAction, QPixmap
from PySide6.QtCore import Qt, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def transform_string_xi_to_x_i_minus_1(input_string):
    """
    Transforms a string like "x2+x3+x101" to "x[1]+x[2]+x[100]".
    """
    def replace_func(match):
        index = int(match.group(1))
        return f"x[{index - 1}]"

    return re.sub(r"x(\d+)", replace_func, input_string)

def add_explicit_multiplication(match):
    """
    Adds explicit multiplication between numbers and variables.
    For example, "2x" becomes "2*x".
    """
    return f"{match.group(1)}*{match.group(2)}"

def make_f_with_step_limit(f_func, max_steps, progress_callback=None):
    counter = {'n': 0}
    def f_wrapped(t, x):
        counter['n'] += 1
        if progress_callback and max_steps:
            percent = int(counter['n'] / max_steps * 100)
            progress_callback(min(percent, 100))
        if max_steps is not None and counter['n'] > max_steps:
            raise RuntimeError(f"Exceeded max steps = {max_steps}")
        return f_func(t, x)
    return f_wrapped, counter

def safe_sqrt(x):
    """
    Safe version of sqrt that handles negative numbers by returning 0.
    """
    if x < 0:
        return 0
    return np.sqrt(x)

def safe_log(x):
    """
    Safe version of log that handles non-positive numbers by returning a large negative number.
    """
    if x <= 0:
        return -1e10
    return np.log(x)

def generate_f_from_equations(equations):
    """
    Generates a function f(t, x) from a list of equations.
    Each equation should be a string representing the right-hand side.
    """
    compiled_eqs = [compile(transform_string_xi_to_x_i_minus_1(eq), '<string>', 'eval') for eq in equations]
    def f(t, x):
        return np.array([eval(expr, {"x": x, "t": t, "np": np, "sqrt": safe_sqrt, "log": safe_log}) for expr in compiled_eqs])
    return f

def generate_f_and_A_from_equations(equations):
    """
    Generates functions f(t, x) and A(t, x) from a list of equations.
    A(t, x) is the Jacobian matrix of f with respect to x.
    """
    # Compile equations
    compiled_eqs = [compile(transform_string_xi_to_x_i_minus_1(eq), '<string>', 'eval') for eq in equations]
    n_vars = len(equations)

    def f(t, x):
        """
        Evaluates the right-hand side of the ODE system.
        """
        return np.array([eval(expr, {"x": x, "t": t, "np": np, "sqrt": safe_sqrt, "log": safe_log}) for expr in compiled_eqs])

    def A(t, x):
        """
        Evaluates the Jacobian matrix of f with respect to x.
        """
        eps = 1e-8
        A = np.zeros((n_vars, n_vars))
        f0 = f(t, x)
        for j in range(n_vars):
            x_eps = x.copy()
            x_eps[j] += eps
            f_eps = f(t, x_eps)
            A[:, j] = (f_eps - f0) / eps
        return A

    def extended_system(t, Y):
        """
        Extended system for [x, vec(X)].
        Y[:n_vars] = x
        Y[n_vars:] = vec(X), where X is the sensitivity matrix.
        """
        x = Y[:n_vars]
        X = Y[n_vars:].reshape(n_vars, n_vars)
        dx = f(t, x)
        dX = np.dot(A(t, x), X)
        return np.concatenate([dx, dX.flatten()])

    return extended_system, n_vars

def parse_term(term_str, t_span, n_vars):
    """
    Parses a single term in a boundary condition.
    Returns the index, whether it's at t=a or t=b, and the value if it's a constant.
    """
    a, b = t_span
    if '(' not in term_str:
        try:
            return None, None, float(term_str)
        except ValueError:
            raise ValueError(f"Invalid term: {term_str}")
    
    name, t_str = term_str.split('(')
    t_val = float(t_str.strip(')'))
    idx = int(name[1:]) - 1
    
    if abs(t_val - a) < 1e-8:
        return idx, True, None
    elif abs(t_val - b) < 1e-8:
        return idx, False, None
    else:
        raise ValueError(f"Invalid t value {t_val}, must be {a} or {b}")

def create_residual_func(lhs_idx, lhs_is_at_a, rhs_idx, rhs_is_at_a, lhs_value, rhs_value, lhs_type, rhs_type):
    """
    Creates a residual function for a boundary condition.
    """
    if lhs_type == 'var' and rhs_type == 'const':
        def residual_func(xa, xb):
            x = xa if lhs_is_at_a else xb
            return x[lhs_idx] - rhs_value
    elif lhs_type == 'const' and rhs_type == 'var':
        def residual_func(xa, xb):
            x = xa if rhs_is_at_a else xb
            return lhs_value - x[rhs_idx]
    elif lhs_type == 'var' and rhs_type == 'var':
        def residual_func(xa, xb):
            x1 = xa if lhs_is_at_a else xb
            x2 = xa if rhs_is_at_a else xb
            return x1[lhs_idx] - x2[rhs_idx]
    else:
        def residual_func(xa, xb):
            return lhs_value - rhs_value
    
    return residual_func

def parse_boundary_conditions_and_jacobians(bc_list, t_span, n_vars):
    """
    Parses boundary conditions and creates functions for computing residuals and their Jacobians.
    """
    a, b = t_span
    residuals = []
    jacobians = []

    for cond in bc_list:
        if '=' not in cond:
            continue
        try:
            left, right = cond.split('=')
            # Parse left side
            if '(' in left:
                name, t_str = left.split('(')
                idx = int(name[1:]) - 1
                t_val = float(t_str.strip(')'))
            else:
                continue

            # Check if right side is a number or reference to another variable
            try:
                val = float(right)
                if abs(t_val - a) < 1e-8:
                    residuals.append(lambda xa, xb, i=idx, v=val: xa[i] - v)
                    jacobians.append((lambda xa, xb, i=idx: np.eye(n_vars)[i], None))
                elif abs(t_val - b) < 1e-8:
                    residuals.append(lambda xa, xb, i=idx, v=val: xb[i] - v)
                    jacobians.append((None, lambda xa, xb, i=idx: np.eye(n_vars)[i]))
                else:
                    raise ValueError(f"Unsupported t {t_val}, must be {a} or {b}")
            except ValueError:
                # If not a number, it's a reference to another variable
                if '(' in right:
                    ref_name, ref_t_str = right.split('(')
                    ref_idx = int(ref_name[1:]) - 1
                    ref_t_val = float(ref_t_str.strip(')'))
                    if abs(t_val - a) < 1e-8 and abs(ref_t_val - a) < 1e-8:
                        residuals.append(lambda xa, xb, i=idx, j=ref_idx: xa[i] - xa[j])
                        jacobians.append((lambda xa, xb, i=idx, j=ref_idx: np.eye(n_vars)[i] - np.eye(n_vars)[j], None))
                    elif abs(t_val - a) < 1e-8 and abs(ref_t_val - b) < 1e-8:
                        residuals.append(lambda xa, xb, i=idx, j=ref_idx: xa[i] - xb[j])
                        jacobians.append((lambda xa, xb, i=idx, j=ref_idx: np.eye(n_vars)[i], lambda xa, xb, i=idx, j=ref_idx: -np.eye(n_vars)[j]))
                    elif abs(t_val - b) < 1e-8 and abs(ref_t_val - a) < 1e-8:
                        residuals.append(lambda xa, xb, i=idx, j=ref_idx: xb[i] - xa[j])
                        jacobians.append((lambda xa, xb, i=idx, j=ref_idx: -np.eye(n_vars)[j], lambda xa, xb, i=idx, j=ref_idx: np.eye(n_vars)[i]))
                    elif abs(t_val - b) < 1e-8 and abs(ref_t_val - b) < 1e-8:
                        residuals.append(lambda xa, xb, i=idx, j=ref_idx: xb[i] - xb[j])
                        jacobians.append((None, lambda xa, xb, i=idx, j=ref_idx: np.eye(n_vars)[i] - np.eye(n_vars)[j]))
                    else:
                        raise ValueError(f"Unsupported t values {t_val} and {ref_t_val}, must be {a} or {b}")
                else:
                    # Special handling for condition x3(1)**2+x4(1)**2=1
                    if "**2" in right:
                        parts = right.split("+")
                        if len(parts) == 2 and "**2" in parts[0] and "**2" in parts[1]:
                            idx1 = int(parts[0].split("(")[0][1:]) - 1
                            idx2 = int(parts[1].split("(")[0][1:]) - 1
                            if abs(t_val - b) < 1e-8:
                                residuals.append(lambda xa, xb, i=idx1, j=idx2: xb[i]**2 + xb[j]**2 - 1)
                                jacobians.append((None, lambda xa, xb, i=idx1, j=idx2: 2*xb[i]*np.eye(n_vars)[i] + 2*xb[j]*np.eye(n_vars)[j]))
                            else:
                                raise ValueError(f"Unsupported t {t_val} for quadratic condition")
        except Exception as e:
            print(f"Error parsing boundary condition '{cond}': {str(e)}")
            continue

    def R(xa, xb):
        """
        Computes the vector of boundary condition residuals.
        """
        return np.array([res(xa, xb) for res in residuals])

    def get_Ra_prime(xa, xb):
        """
        Computes the Jacobian of R with respect to xa.
        """
        Ra_prime = np.zeros((len(residuals), n_vars))
        for i, (Ra, _) in enumerate(jacobians):
            if Ra is not None:
                Ra_prime[i] = Ra(xa, xb)
        return Ra_prime

    def get_Rb_prime(xa, xb):
        """
        Computes the Jacobian of R with respect to xb.
        """
        Rb_prime = np.zeros((len(residuals), n_vars))
        for i, (_, Rb) in enumerate(jacobians):
            if Rb is not None:
                Rb_prime[i] = Rb(xa, xb)
        return Rb_prime

    return R, get_Ra_prime, get_Rb_prime, len(residuals)

def solve_ode(t_span, p, tol, method, # method - это method_inner
              f_system, # Это исходная f(t, x)
              t_eval=None, # Разрешаем прямое указание t_eval
              max_step=None, max_steps=None):
    """
    Решает исходную систему ОДУ dx/dt = f(t, x) при заданном начальном условии x(a) = p.
    Использует scipy.integrate.solve_ivp с указанным методом.
    """
    t0, t1 = t_span
    # Если t_eval не предоставлен, генерируем сетку для построения графиков
    if t_eval is None:
        t_eval = np.linspace(t0, t1, 500) # Сетка по умолчанию для построения графиков

    f_wrapped, counter = make_f_with_step_limit(f_system, max_steps)
    ivp_kwargs = dict(rtol=tol, atol=tol*1e-2, method=method, dense_output=True)
    # Всегда запрашиваем dense_output=True для вычисления решения в произвольных точках
    ivp_kwargs['dense_output'] = True
    ivp_kwargs['t_eval'] = t_eval # Передаем потенциально сгенерированный t_eval

    if max_step is not None:
        ivp_kwargs['max_step'] = max_step
    sol = solve_ivp(f_wrapped, t_span, p, **ivp_kwargs)
    if not sol.success:
         raise RuntimeError(f"Решатель ОДУ завершился с ошибкой: {sol.message}")
    return sol.sol, counter['n']

def solve_ode_extended(t_span, x_initial, n_vars, tol, method, # method - это method_inner
                       extended_system, # Это расширенная система для [x, vec(X)]
                       N_mesh=None, max_step=None, max_steps=None):
    """
    Решает расширенную систему ОДУ [x(t), vec(X(t))] при заданном начальном условии x(a) = x_initial.
    Начальное условие для X(a) - единичная матрица.
    Использует scipy.integrate.solve_ivp с указанным методом.
    """
    t0, t1 = t_span
    # Нам нужен плотный вывод для расширенной системы для вычисления X(b)
    t_eval = np.linspace(t0, t1, N_mesh) if N_mesh else None # Сохраняем опцию N_mesh, если нужна внутри

    # Начальное условие для расширенной системы Y = [x(a), vec(I)]
    n_total_vars = n_vars + n_vars * n_vars
    y0 = np.zeros(n_total_vars)
    y0[:n_vars] = x_initial # Начальное условие для x
    y0[n_vars:] = np.eye(n_vars).flatten() # Начальное условие для X (единичная матрица, сплющенная)

    f_wrapped, counter = make_f_with_step_limit(extended_system, max_steps)
    ivp_kwargs = dict(rtol=tol, atol=tol*1e-2, method=method, dense_output=True)
    if t_eval is not None:
        ivp_kwargs['t_eval'] = t_eval
    if max_step is not None:
        ivp_kwargs['max_step'] = max_step

    sol = solve_ivp(f_wrapped, t_span, y0, **ivp_kwargs)

    if not sol.success:
         raise RuntimeError(f"Решатель ОДУ завершился с ошибкой: {sol.message}")

    # Возвращаем функцию, которая предоставляет [x(t), X(t)]
    def sol_func(t):
        y_at_t = sol.sol(t)
        x_at_t = y_at_t[:n_vars]
        X_at_t = y_at_t[n_vars:].reshape((n_vars, n_vars))
        return x_at_t, X_at_t

    return sol_func, counter['n']

def phi(p, t_span, tol_inner, method_inner, # передаем method_inner сюда
        extended_system, n_vars,
        N_mesh=None, max_step=None, max_steps=None,
        R_func=None, get_Ra_prime=None, get_Rb_prime=None):
    """
    Вычисляет значение функции стрельбы R(x(a,p), x(b,p)) и ее Якобиан Phi'(p).
    p - начальное предположение для x(a).
    """
    # Решаем расширенную систему, начиная с x(a) = p и X(a) = I
    sol_extended, _ = solve_ode_extended(t_span, p, n_vars, tol_inner, method_inner, extended_system, # Используем method_inner
                             N_mesh, max_step, max_steps)

    # Получаем решение на границах
    xa, Xa = sol_extended(t_span[0]) # x(a), X(a). Xa должно быть близко к единичной матрице, если p точно равно x(a)
    xb, Xb = sol_extended(t_span[1]) # x(b), X(b)

    # Вычисляем R(x(a), x(b)) - значение функции стрельбы (невязка)
    phi_val = R_func(xa, xb)

    # Вычисляем Якобианы краевых условий по x(a) и x(b)
    Ra_prime = get_Ra_prime(xa, xb)
    Rb_prime = get_Rb_prime(xa, xb)

    # Вычисляем Phi'(p) = Ra' * d(xa)/dp + Rb' * d(xb)/dp
    # Поскольку p определено как начальное условие x(a), d(xa)/dp = I (Единичная матрица)
    # А d(xb)/dp = X(b, p), матрица чувствительности, вычисленная при t=b.
    dphi_dp_val = np.dot(Ra_prime, np.eye(n_vars)) + np.dot(Rb_prime, Xb)

    return phi_val, dphi_dp_val # Возвращаем как значение функции, так и ее Якобиан

def solve_corrector_equation(p_guess, mu, t_span, tol_inner, method_inner, # Передаем method_inner
                             extended_system, n_vars, R_func, get_Ra_prime, get_Rb_prime,
                             initial_phi_at_p0, # Постоянная невязка Phi(p0_initial)
                             corrector_method='hybr', # Метод для scipy.optimize.root
                             tol_corrector=1e-6, max_corrector_steps=10):
    """
    Решает Phi(p) - (1-mu)*Phi(p0) = 0 для p при заданном mu с использованием scipy.optimize.root.
    Позволяет выбрать метод для поиска корня.
    """
    # Определяем функцию, корень которой мы ищем: Psi(p, mu)
    def psi_func(p_current):
        # Используем существующую функцию phi для получения Phi(p_current)
        # method_inner требуется здесь для внутреннего решения ОДУ внутри phi
        phi_val, _ = phi(p_current, t_span, tol_inner, method_inner, extended_system, n_vars,
                         R_func=R_func, get_Ra_prime=get_Ra_prime, get_Rb_prime=get_Rb_prime)
        return phi_val - (1.0 - mu) * initial_phi_at_p0

    # Определяем Якобиан Psi(p, mu) по p: dPsi/dp = Phi'(p)
    # Это требуется только для методов, использующих Якобиан (например, 'krylov', 'hybr', 'lm')
    def psi_jac(p_current):
         # Используем существующую функцию phi для получения Phi'(p_current)
         # method_inner требуется здесь для внутреннего решения ОДУ внутри phi
        _, J = phi(p_current, t_span, tol_inner, method_inner, extended_system, n_vars,
                   R_func=R_func, get_Ra_prime=get_Ra_prime, get_Rb_prime=get_Rb_prime)
        return J # Phi'(p) - это Якобиан Psi(p, mu) по p

    # Вызываем scipy.optimize.root
    # Передаем функцию Якобиана, если метод ее требует или может использовать
    # 'hybr' и 'lm' могут использовать Якобиан, если он предоставлен, 'krylov' требует его.
    # Методы Бройдена не используют аналитический Якобиан.
    jac_arg = psi_jac if corrector_method in ['krylov', 'hybr', 'lm'] else None

    # Опции для поиска корня
    options = {'maxiter': max_corrector_steps}
    # Примечание: 'tol' в root обычно относится к норме значения функции (невязки)
    # Некоторые методы могут интерпретировать опции по-разному.
    # Для 'krylov', 'tol' в options относится к размеру шага, а не к невязке.
    # Основной аргумент 'tol' функции root относится к норме невязки.

    try:
        result = root(psi_func, p_guess, jac=jac_arg, method=corrector_method, tol=tol_corrector, options=options)

        if not result.success:
            # Выводим предупреждение, но не обязательно сразу вызываем ошибку,
            # так как продолжение может восстановиться на следующем шаге.
            print(f"  Предупреждение: Решение внешней задачи ({corrector_method}) не сошлось при mu={mu}: {result.message}")
            # Выводим норму невязки для отладки
            print(f"  Норма невязки при mu={mu}: {np.linalg.norm(result.fun):.2e}")
            # В зависимости от желаемой строгости, можно раскомментировать следующую строку:
            # raise RuntimeError(f"Решение внешней задачи не сошлось при mu={mu}: {result.message}")


    except ValueError as e:
        # Перехватываем потенциальные ошибки из scipy.optimize.root (например, проверка входных данных)
         print(f"  Ошибка вызова scipy.optimize.root при mu={mu} с методом '{corrector_method}': {e}")
         raise RuntimeError(f"Решение внешней задачи завершилось с ошибкой при mu={mu}: {e}")

    except Exception as e:
         # Перехватываем другие потенциальные ошибки во время решения внешней задачи (например, из вызова phi)
         print(f"  Произошла непредвиденная ошибка во время решения внешней задачи при mu={mu} с методом '{corrector_method}': {e}")
         raise RuntimeError(f"Решение внешней задачи завершилось с ошибкой при mu={mu}: {e}")


    return result.x # Найденный корень - это скорректированный вектор параметра p

def continuation_method(p0_initial, t_span, mu_span,
                        tol_inner, tol_corrector,
                        method_inner,
                        f_system_equations, bc_list,
                        corrector_method='hybr',
                        n_mu_steps=100,
                        max_corrector_steps=10,
                        N_mesh_inner=None,
                        max_step_inner=None,
                        max_steps_inner=None,
                        grid_points_for_output=500,
                        progress_callback=None # Добавляем callback для прогресса
                       ):
    """
    Решает КЗ методом продолжения с использованием схемы предиктор-решение внешней задачи.
    Отслеживает решение p(mu) для гомотопии Phi(p) - (1-mu)*Phi(p0) = 0.
    Позволяет выбрать метод для внутреннего решателя ОДУ и решения внешней задачи.
    Включает callback для обновления прогресса.
    """
    mu_start, mu_end = mu_span
    mu_steps = np.linspace(mu_start, mu_end, n_mu_steps + 1)
    dmu = (mu_end - mu_start) / n_mu_steps

    # 1. Определяем расширенную систему ОДУ [x, vec(X)] и получаем размерность задачи
    extended_system, n_vars = generate_f_and_A_from_equations(f_system_equations)

    # 2. Парсим краевые условия и получаем функции для R, Ra', Rb'
    R_func, get_Ra_prime, get_Rb_prime, n_bc = parse_boundary_conditions_and_jacobians(bc_list, t_span, n_vars)

    if n_vars != n_bc:
        print(f"Предупреждение: Количество переменных состояния ({n_vars}) не совпадает с количеством краевых условий ({n_bc}).")


    # 3. Вычисляем начальную невязку Phi(p0_initial). Этот член постоянен
    # на протяжении всего процесса продолжения.
    print(f"Вычисляем начальную невязку Phi(p0) при p0 = {p0_initial} (при mu={mu_start})...")
    initial_phi_at_p0, initial_J_at_p0 = phi(p0_initial, t_span, tol_inner, method_inner, extended_system, n_vars,
                                            N_mesh_inner, max_step_inner, max_steps_inner,
                                            R_func, get_Ra_prime, get_Rb_prime)
    print(f"Начальная невязка Phi(p0) = {initial_phi_at_p0}")


    # Инициализируем текущее решение начальным предположением
    p_current = p0_initial.copy()

    # Храним путь p во время продолжения (опционально)
    p_path = [p_current.copy()]
    mu_path = [mu_start]

    # 4. Цикл продолжения (Предиктор-Решение внешней задачи)
    print(f"Начинаем продолжение от mu={mu_start} до mu={mu_end} с {n_mu_steps} шагами...")
    for i in range(n_mu_steps):
        mu_k = mu_steps[i]
        mu_k_plus_1 = mu_steps[i+1]

        print(f"  Обработка шага mu {i+1}/{n_mu_steps} (mu={mu_k_plus_1:.2f})")

        # Обновляем прогресс-бар
        if progress_callback:
            progress_callback(i + 1)


        # Шаг предиктора: Используем касательную в p_current для оценки решения при mu_k_plus_1
        # Касательная dp/dmu = -[Phi'(p_current)]^-1 * initial_phi_at_p0
        try:
            # Вычисляем Phi'(p_current), решая расширенную систему
            _, J_current = phi(p_current, t_span, tol_inner, method_inner, extended_system, n_vars,
                               N_mesh=N_mesh_inner, max_step=max_step_inner, max_steps=max_steps_inner, # Передаем опции внутреннего решателя
                               R_func=R_func, get_Ra_prime=get_Ra_prime, get_Rb_prime=get_Rb_prime)

            # Решаем J_current * касательная = -initial_phi_at_p0 для касательной
            tangent = -np.linalg.solve(J_current, initial_phi_at_p0)
        except np.linalg.LinAlgError:
            print(f"Предупреждение: Якобиан предиктора сингулярен при mu={mu_k}. Используем псевдообратную матрицу.")
            tangent = -np.linalg.pinv(J_current) @ initial_phi_at_p0
        except RuntimeError as e:
             print(f"Ошибка во время шага предиктора при mu={mu_k} (внутреннее решение ОДУ): {e}")
             raise # Повторно вызываем ошибку

        p_predicted = p_current + tangent * dmu

        # Шаг решения внешней задачи: Используем поиск корня (scipy.optimize.root) для нахождения решения
        # при mu_k_plus_1, начиная с предсказанного значения.
        try:
            p_corrected = solve_corrector_equation(
                p_predicted, mu_k_plus_1, t_span, tol_inner, method_inner,
                extended_system, n_vars, R_func, get_Ra_prime, get_Rb_prime,
                initial_phi_at_p0,
                corrector_method=corrector_method,
                tol_corrector=tol_corrector,
                max_corrector_steps=max_corrector_steps
            )
        except RuntimeError as e:
             print(f"Ошибка во время шага решения внешней задачи при mu={mu_k_plus_1}: {e}")
             raise # Повторно вызываем ошибку


        # Обновляем текущее решение для следующего шага
        p_current = p_corrected

        # Сохраняем скорректированное решение (опционально)
        p_path.append(p_current.copy())
        mu_path.append(mu_k_plus_1)

        # Выводим прогресс
        if (i + 1) % (n_mu_steps // 10) == 0 or (i + 1) == n_mu_steps:
             phi_at_corrected, _ = phi(p_current, t_span, tol_inner, method_inner, extended_system, n_vars,
                                       N_mesh=N_mesh_inner, max_step=max_step_inner, max_steps=max_steps_inner,
                                       R_func=R_func, get_Ra_prime=get_Ra_prime, get_Rb_prime=get_Rb_prime)
             residual_norm_at_mu1 = np.linalg.norm(phi_at_corrected)
             print(f"  Шаг mu {i+1}/{n_mu_steps} (mu={mu_k_plus_1:.2f}) завершен. Норма невязки при mu=1: {residual_norm_at_mu1:.2e}")


    print("Процесс продолжения завершен.")


    # Окончательное решение - это p_current после последнего шага
    final_p = p_current

    # Проверяем, выполняются ли краевые условия для окончательного решения
    # Решаем исходную ОДУ с окончательными начальными условиями, чтобы получить решение при t=t_span[1]
    f_original_final = generate_f_from_equations(f_system_equations)

    # Генерируем точки времени для выходной сетки
    t_eval_output = np.linspace(t_span[0], t_span[1], grid_points_for_output)

    # Решаем ОДУ с окончательным p, чтобы получить траекторию, вычисленную в точках выходной сетки
    sol_func_final_p, _ = solve_ode(t_span, final_p, tol_inner, method_inner, f_original_final,
                                     t_eval=t_eval_output) # Передаем точки выходной сетки

    # Значения решения в точках сетки доступны непосредственно из sol_func_final_p(t_eval_output)
    x_at_grid_points = sol_func_final_p(t_eval_output)


    x_at_a_final = sol_func_final_p(t_span[0])
    x_at_b_final = sol_func_final_p(t_span[1])
    residual_at_final_p = R_func(x_at_a_final, x_at_b_final)
    print(f"Окончательная норма невязки для целевой задачи (mu={mu_end}): {np.linalg.norm(residual_at_final_p):.2e}")

    # Возвращаем окончательный параметр p, функцию для получения траектории и решение в точках сетки
    return final_p, sol_func_final_p, t_eval_output, x_at_grid_points

def plot_trajectory(sol_func, t_span, num_points=500, title=None):
    t_vals = np.linspace(*t_span, num_points)
    xs = sol_func(t_vals)
    plt.figure()
    plt.plot(xs[0], xs[1], '-o', markersize=2)
    if title:
        plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()

def plot_custom_trajectory(sol_func, t_span, x_idx=0, y_idx=1, num_points=500, title=None):
    t_vals = np.linspace(*t_span, num_points)
    xs = sol_func(t_vals)
    x_data = t_vals if x_idx == -1 else xs[x_idx]
    y_data = t_vals if y_idx == -1 else xs[y_idx]

    plt.figure()
    plt.plot(x_data, y_data, '-o', markersize=2)
    if title:
        plt.title(title)
    plt.xlabel(f"{'t' if x_idx == -1 else f'x{x_idx+1}'}")
    plt.ylabel(f"{'t' if y_idx == -1 else f'x{y_idx+1}'}")
    plt.grid(True)
    plt.show()



def plot_phase_portrait(sol_func, t_span, num_points=500, title=None):
    plot_trajectory(sol_func, t_span, num_points, title)

    
    
import sys
import json
import numpy as np
from scipy.integrate import solve_bvp
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QComboBox,
    QHBoxLayout, QGridLayout, QTextEdit, QDialog, QSpinBox, QFrame, QStyleFactory,
    QSplitter, QScrollArea, QGroupBox
)
from PySide6.QtGui import QKeySequence, QPalette, QColor, QFont, QIcon
from PySide6.QtCore import Qt, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        # Restore default matplotlib style (white background, black text)
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        self.ax.grid(True)
        self.ax.spines['top'].set_visible(True)
        self.ax.spines['right'].set_visible(True)
        self.ax.tick_params(colors='black')
        self.ax.xaxis.label.set_color('black')
        self.ax.yaxis.label.set_color('black')

    def plot(self, x, y):
        self.ax.clear()
        self.ax.set_facecolor('white')
        
        # Проверяем и преобразуем размерности
        x = np.asarray(x)
        y = np.asarray(y)
        
        print(f"Plot input shapes - x: {x.shape}, y: {y.shape}")
        
        # Если y одномерный, преобразуем его в двумерный
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        
        # Если размерности не совпадают, транспонируем y
        if y.shape[1] != len(x):
            y = y.T
        
        print(f"Plot processed shapes - x: {x.shape}, y: {y.shape}")
        
        # Строим график для каждой переменной
        for i in range(y.shape[0]):
            self.ax.plot(x, y[i], label=f'x{i+1}')
            
        self.ax.grid(True)
        self.ax.tick_params(colors='black')
        self.ax.xaxis.label.set_color('black')
        self.ax.yaxis.label.set_color('black')
        self.ax.legend()
        self.draw()

class ResultsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Результаты задачи")
        self.setMinimumSize(800, 600)
        
        # Настройка стиля
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f6f7;
            }
            QLabel {
                color: black;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QComboBox {
                padding: 7px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                color: black;
                font-size: 14px;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 7px;
                color: black;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #34495e;
                border-radius: 4px;
                margin-top: 10px;
                color: black;
                font-weight: bold;
                font-size: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: black;
                font-size: 15px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Создаем сплиттер для разделения окна
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель с графиком
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        self.plot = PlotWidget()
        left_layout.addWidget(self.plot)
        
        # Кнопка интеграла под графиком
        self.integral_btn = QPushButton("Интеграл")
        self.integral_btn.setStyleSheet("background-color: #27ae60; color: white; font-size: 16px; padding: 8px 20px; border-radius: 5px;")
        self.integral_btn.clicked.connect(self.show_integral_dialog)
        left_layout.addWidget(self.integral_btn)
        
        # Группа для выбора осей
        axis_group = QGroupBox("Настройки графика")
        axis_layout = QVBoxLayout()
        
        # X ось
        x_frame = QFrame()
        x_layout = QHBoxLayout(x_frame)
        x_layout.addWidget(QLabel("Ось X:"))
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(["t", "x1", "x2", "x3", "x4"])
        x_layout.addWidget(self.x_axis_combo)
        axis_layout.addWidget(x_frame)
        
        # Y ось (в одной строке с полем)
        y_frame = QFrame()
        y_layout = QHBoxLayout(y_frame)
        y_layout.addWidget(QLabel("Ось Y:"))
        self.y_expr_edit = QTextEdit()
        self.y_expr_edit.setPlaceholderText("Введите выражения для оси Y, по одному на строку, например:\nx1\nx2\nx1+x2\nnp.sin(x1)")
        self.y_expr_edit.setFixedHeight(60)
        self.y_expr_edit.setFixedWidth(220)
        y_layout.addWidget(self.y_expr_edit)
        axis_layout.addWidget(y_frame)
        # --- Автоматическое обновление графика при изменении выражения ---
        self.x_axis_combo.currentIndexChanged.connect(self.plot_selected_axes)
        self.y_expr_edit.textChanged.connect(self.plot_selected_axes)
        
        axis_group.setLayout(axis_layout)
        left_layout.addWidget(axis_group)
        
        splitter.addWidget(left_panel)
        
        # Правая панель с результатами
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        results_group = QGroupBox("Результаты")
        results_layout = QVBoxLayout()
        self.result_table = QTextEdit()
        self.result_table.setReadOnly(True)
        results_layout.addWidget(self.result_table)
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)
        
        splitter.addWidget(right_panel)
        
        layout.addWidget(splitter)
        
        # Кнопка закрытия
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)

        self.last_sol = None
        self.last_t_span = None

    def set_task_title(self, task_name):
        """Sets the task name in the window title."""
        if task_name:
            self.setWindowTitle(f"Результаты задачи: {task_name}")
        else:
            self.setWindowTitle("Результаты задачи")

    def display_results(self, x, y, sol_func=None, t_span=None):
        # Сохраняем решение и временной интервал
        self.last_sol = sol_func
        self.last_t_span = t_span
        
        # Проверяем размерности и преобразуем массивы при необходимости
        x = np.asarray(x)
        y = np.asarray(y)
        print(f"Display results input - x shape: {x.shape}, y shape: {y.shape}")
        
        # Если y одномерный, преобразуем его в двумерный
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        # Если y двумерный и размерности не совпадают, транспонируем
        elif len(y.shape) == 2 and y.shape[1] != len(x):
            y = y.T
            
        print(f"Display results after reshape - x shape: {x.shape}, y shape: {y.shape}")
        
        # Строим график
        self.plot.plot(x, y)
        
        # Создаем таблицу для результатов
        table_text = """
        <table border='1' cellpadding='3' style='border-collapse: collapse; width: auto; min-width: 100%; font-size: 12px; font-family: monospace;'>
        """
        
        # Заголовок таблицы
        table_text += "<tr style='background-color: #f0f0f0;'>"
        table_text += "<th style='min-width: 90px; white-space: nowrap;'>t</th>"
        for i in range(y.shape[0]):
            table_text += f"<th style='min-width: 130px; white-space: nowrap;'>x{i+1}</th>"
        table_text += "</tr>"
        
        # Получаем значение t* и шаг из главного окна
        main_window = self.parent()
        if main_window and hasattr(main_window, 't_star_input'):
            t_star_text = main_window.t_star_input.text()
            step_text = main_window.step_input.text()
            
            try:
                step = float(step_text) if step_text else 0.1  # По умолчанию шаг 0.1
            except ValueError:
                step = 0.1  # Если шаг введен некорректно, используем значение по умолчанию
            
            if t_star_text:
                try:
                    t_star = float(t_star_text)
                    # Находим ближайшую точку к t*
                    idx = np.abs(x - t_star).argmin()
                    
                    # Добавляем строку с результатами для t*
                    table_text += "<tr>"
                    table_text += f"<td style='white-space: nowrap;'>{t_star:.3f}</td>"
                    for val in y[:, idx]:
                        table_text += f"<td style='white-space: nowrap;'>{val:.3f}</td>"
                    table_text += "</tr>"
                except ValueError:
                    # Если t* введен некорректно, показываем точки с заданным шагом по времени
                    t_start = x[0]
                    t_end = x[-1]
                    t_values = np.arange(t_start, t_end + step/2, step)  # +step/2 для учета погрешности округления
                    
                    for t in t_values:
                        # Находим ближайшую точку к текущему t
                        idx = np.abs(x - t).argmin()
                        if abs(x[idx] - t) < step/2:  # Проверяем, что точка достаточно близка
                            table_text += "<tr>"
                            table_text += f"<td style='white-space: nowrap;'>{t:.3f}</td>"
                            for val in y[:, idx]:
                                table_text += f"<td style='white-space: nowrap;'>{val:.3f}</td>"
                            table_text += "</tr>"
            else:
                # Если t* не введен, показываем точки с заданным шагом по времени
                t_start = x[0]
                t_end = x[-1]
                t_values = np.arange(t_start, t_end + step/2, step)  # +step/2 для учета погрешности округления
                
                for t in t_values:
                    # Находим ближайшую точку к текущему t
                    idx = np.abs(x - t).argmin()
                    if abs(x[idx] - t) < step/2:  # Проверяем, что точка достаточно близка
                        table_text += "<tr>"
                        table_text += f"<td style='white-space: nowrap;'>{t:.3f}</td>"
                        for val in y[:, idx]:
                            table_text += f"<td style='white-space: nowrap;'>{val:.3f}</td>"
                        table_text += "</tr>"
        else:
            # Если не удалось получить значения, показываем точки с шагом по умолчанию
            t_start = x[0]
            t_end = x[-1]
            t_values = np.arange(t_start, t_end + 0.1/2, 0.1)  # Шаг по умолчанию 0.1
            
            for t in t_values:
                idx = np.abs(x - t).argmin()
                if abs(x[idx] - t) < 0.1/2:
                    table_text += "<tr>"
                    table_text += f"<td style='white-space: nowrap;'>{t:.3f}</td>"
                    for val in y[:, idx]:
                        table_text += f"<td style='white-space: nowrap;'>{val:.3f}</td>"
                    table_text += "</tr>"
        
        table_text += "</table>"
        
        # Обернем таблицу в div с горизонтальной прокруткой
        scrollable_table = f"""
        <div style='width: 100%; overflow-x: auto;'>
            {table_text}
        </div>
        """
        
        self.result_table.setHtml(scrollable_table)
        
        # Создаем функцию-обертку для решения
        def wrapped_sol_func(t):
            if isinstance(t, np.ndarray):
                return np.array([sol_func(ti) for ti in t])
            return sol_func(t)
        
        # Сохраняем обернутую функцию решения
        self.last_sol = wrapped_sol_func
        
        # Обновляем график по текущим выражениям Y
        self.plot_selected_axes()

    def update_axis_selectors(self, n_vars):
        # Only add variables that exist in the solution
        items = ["t"] + [f"x{i+1}" for i in range(n_vars)]
        self.x_axis_combo.clear()
        self.x_axis_combo.addItems(items)


    def plot_selected_axes(self):
        if self.last_sol is None or self.last_t_span is None:
            print("No solution available")
            return
            
        # Get time values
        t_vals = np.linspace(*self.last_t_span, 500)
        print(f"Time values shape: {t_vals.shape}")
        
        try:
            # Evaluate solution at time points
            xs = []
            for t in t_vals:
                x = self.last_sol(t)  # Получаем только x
                if isinstance(x, tuple):  # Если решение возвращает кортеж
                    x = x[0]  # Берем только первую часть (x)
                print(f"Solution at t={t}: shape={x.shape if hasattr(x, 'shape') else 'scalar'}, value={x}")
                xs.append(x)
            xs = np.array(xs)  # shape: (n_points, n_vars)
            print(f"Solution array shape before transpose: {xs.shape}")
            xs = xs.T  # shape: (n_vars, n_points)
            print(f"Solution array shape after transpose: {xs.shape}")
        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return
            
        # Get number of variables from solution shape
        n_vars = xs.shape[0]
        print(f"Number of variables: {n_vars}")
        
        # Setup axis mapping
        axis_names = ["t"] + [f"x{i+1}" for i in range(n_vars)]
        axis_map = {name: i - 1 for i, name in enumerate(axis_names)}
        print(f"Axis mapping: {axis_map}")
        
        # Get selected X axis
        x_name = self.x_axis_combo.currentText()
        x_idx = axis_map.get(x_name, -1)
        print(f"Selected X axis: {x_name}, index: {x_idx}")
        x_data = t_vals if x_idx == -1 else xs[x_idx]
        print(f"X data shape: {x_data.shape}")
        
        # Get expressions to plot
        exprs = [line.strip() for line in self.y_expr_edit.toPlainText().splitlines() if line.strip()]
        print(f"Expressions to evaluate: {exprs}")
        
        y_curves = []
        legends = []
        
        for expr in exprs:
            y_data = []
            error = False
            
            for i, t in enumerate(t_vals):
                local_vars = {"t": t, "np": np}
                # Add all variables to local context
                for j in range(n_vars):
                    local_vars[f"x{j+1}"] = xs[j, i]
                
                try:
                    y_val = eval(expr, local_vars)
                    y_data.append(y_val)
                except Exception as e:
                    print(f"Error evaluating expression '{expr}': {e}")
                    error = True
                    break
            
            if not error and y_data:
                y_curves.append(np.array(y_data))
                legends.append(expr)
        
        # Plot the curves
        self.plot.ax.clear()
        if y_curves:
            # Преобразуем список кривых в двумерный массив
            y_data = np.array(y_curves)  # shape: (n_curves, n_points)
            print(f"Final y_data shape: {y_data.shape}")
            
            # Проверяем размерности
            if len(x_data) != y_data.shape[1]:
                print(f"Warning: x_data length ({len(x_data)}) != y_data columns ({y_data.shape[1]})")
                # Обрезаем массивы до минимальной длины
                min_len = min(len(x_data), y_data.shape[1])
                x_data = x_data[:min_len]
                y_data = y_data[:, :min_len]
            
            for i, (y, label) in enumerate(zip(y_data, legends)):
                self.plot.ax.plot(x_data, y, label=label)
            self.plot.ax.grid(True)
            self.plot.ax.legend()
        else:
            self.plot.ax.plot(x_data, np.zeros_like(x_data))
            
        self.plot.draw()

    def show_integral_dialog(self):
        dialog = IntegralDialog(self)
        dialog.exec_()

    def plot_quick_response(self):
        """Special plotting function for quick response task"""
        if self.last_sol is None or self.last_t_span is None:
            print("No solution available")
            return
            
        # Get time values
        t_vals = np.linspace(*self.last_t_span, 500)
        
        try:
            # Evaluate solution at time points
            xs = self.last_sol(t_vals)
            print(f"Solution shape: {xs.shape}")
        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return
            
        # Get expressions to plot
        exprs = [line.strip() for line in self.y_expr_edit.toPlainText().splitlines() if line.strip()]
        print(f"Expressions to evaluate: {exprs}")
        
        y_curves = []
        legends = []
        
        for expr in exprs:
            y_data = []
            error = False
            
            for i, t in enumerate(t_vals):
                local_vars = {"t": t, "np": np}
                # For quick response task, we have u1 and u2
                local_vars["u1"] = xs[0, i]
                local_vars["u2"] = xs[1, i]
                
                try:
                    y_val = eval(expr, local_vars)
                    y_data.append(y_val)
                except Exception as e:
                    print(f"Error evaluating expression '{expr}': {e}")
                    error = True
                    break
            
            if not error and y_data:
                y_curves.append(np.array(y_data))
                legends.append(expr)
        
        # Plot the curves
        self.plot.ax.clear()
        if y_curves:
            for y, label in zip(y_curves, legends):
                self.plot.ax.plot(t_vals, y, label=label)
            self.plot.ax.grid(True)
            self.plot.ax.legend()
        else:
            self.plot.ax.plot(t_vals, np.zeros_like(t_vals))
            
        self.plot.draw()

    def show_quick_response_plots(self):
        import numpy as np
        
        # Создаем данные для графиков
        t = np.linspace(0, 1, 500)
        
        # Данные для u1
        t_a, t_b = 0.0, 0.15
        t_c, t_d = 0.45, 0.55
        u1 = np.piecewise(t,
            [(t > t_a) & (t < t_b),
             (t >= t_b) & (t < t_c),
             (t >= t_c) & (t < t_d),
             t >= t_d],
            [lambda t: -0.5 * np.sin(np.pi * (t - t_a) / (t_b - t_a)),
             0,
             lambda t: 0.5 * np.sin(np.pi * (t - t_c) / (t_d - t_c)),
             0])
        
        # Данные для u2
        t1, t2, t3, t4 = 0.2, 0.5, 0.8, 1.0
        u2 = np.piecewise(t,
            [t < t1,
             (t >= t1) & (t < t2),
             (t >= t2) & (t < t3),
             t >= t3],
            [lambda t: (t - 0)/(t1 - 0)*1,
             1,
             lambda t: 1 + (t - t2)/(t3 - t2)*(-2),
             -1])
        
        # Создаем массив для отображения в ResultsWindow
        y = np.vstack((u1, u2))
        
        # Создаем функцию-обертку для решения
        def sol_func(t):
            if isinstance(t, np.ndarray):
                return np.array([np.array([u1[i], u2[i]]) for i in range(len(t))])
            idx = np.abs(t - t).argmin()
            return np.array([u1[idx], u2[idx]])
        
        # Обновляем окно результатов
        self.results.update_axis_selectors(2)  # 2 переменные управления
        self.results.set_task_title(self.task_name_input.text())
        self.results.display_results(t, y, sol_func, (0, 1))
        
        # Настраиваем график
        self.results.plot.ax.clear()
        self.results.plot.ax.plot(t, u1, '-', linewidth=2, label='$u_1(t)$')
        self.results.plot.ax.plot(t, u2, '--', linewidth=2, label='$u_2(t)$')
        
        # Добавляем маркеры
        self.results.plot.ax.plot(0, 0, marker='o', markerfacecolor='white', markeredgecolor='black')
        self.results.plot.ax.plot(t_b, 0, marker='o', markerfacecolor='white', markeredgecolor='black')
        self.results.plot.ax.plot(t_c, 0, marker='o', markerfacecolor='white', markeredgecolor='black')
        self.results.plot.ax.plot(t_d, 0, marker='o', markerfacecolor='white', markeredgecolor='black')
        self.results.plot.ax.plot(1, -1, marker='o', markerfacecolor='white', markeredgecolor='black')
        
        # Настраиваем оси и сетку
        self.results.plot.ax.set_ylim(-1.5, 1.5)
        self.results.plot.ax.set_xlim(-0.05, 1.05)
        self.results.plot.ax.set_xlabel('$t$')
        self.results.plot.ax.set_ylabel('$u(t)$')
        self.results.plot.ax.grid(True)
        self.results.plot.ax.legend()
        
        # Обновляем отображение
        self.results.plot.draw()
        
        # Показываем окно результатов
        self.results.exec_()


class IntegralDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Нахождение интеграла")
        self.setMinimumSize(600, 250)
        layout = QVBoxLayout()
        grid = QGridLayout()
        grid.setSpacing(10)

        # Верхний предел
        self.upper_edit = QLineEdit("1")
        self.upper_edit.setFixedWidth(60)
        grid.addWidget(self.upper_edit, 0, 1)

        # Нижний предел
        self.lower_edit = QLineEdit("0")
        self.lower_edit.setFixedWidth(60)
        grid.addWidget(self.lower_edit, 2, 1)

        # Символ интеграла
        int_label = QLabel("<span style='font-size:48pt;'>&#8747;</span>")
        int_label.setAlignment(Qt.AlignCenter)
        grid.addWidget(int_label, 1, 0, 1, 1)

        # Поле для функции
        self.func_edit = QLineEdit()
        self.func_edit.setPlaceholderText("Введите функцию от t, x1, x2, ...")
        grid.addWidget(self.func_edit, 1, 1, 1, 3)

        # dt
        dt_label = QLabel("<span style='font-size:24pt;'>dt</span>")
        grid.addWidget(dt_label, 1, 4)

        # Кнопка вычисления
        self.calc_btn = QPushButton("=")
        self.calc_btn.setFixedWidth(60)
        self.calc_btn.clicked.connect(self.calculate_integral)
        grid.addWidget(self.calc_btn, 1, 5)

        # Результат
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("font-size: 18pt; color: #2c3e50;")
        grid.addWidget(self.result_label, 1, 6)

        layout.addLayout(grid)

        # Кнопка закрытия
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        self.setLayout(layout)

    def calculate_integral(self):
        try:
            a = float(self.lower_edit.text())
            b = float(self.upper_edit.text())
            func_str = self.func_edit.text().replace('^', '**')
            # Определяем допустимые переменные: t, x1, x2, ... по числу строк в boundary_input или equation_input
            allowed_vars = {"t"}
            n_vars = 0
            # Получаем MainWindow через ResultsWindow
            mainwin = None
            parent = self.parent()
            if parent and hasattr(parent, 'parent'):
                mainwin = parent.parent()
            if mainwin and hasattr(mainwin, 'boundary_input'):
                n_vars = mainwin.boundary_input.rowCount()
                for i in range(n_vars):
                    allowed_vars.add(f"x{i+1}")
            elif mainwin and hasattr(mainwin, 'equation_input'):
                n_vars = mainwin.equation_input.rowCount()
                for i in range(n_vars):
                    allowed_vars.add(f"x{i+1}")
            # Проверяем, что используются только разрешённые переменные
            import re
            used_vars = set(re.findall(r'\b[a-zA-Z_]\w*\b', func_str))
            for v in used_vars:
                if v not in allowed_vars and v not in {"np", "sin", "cos", "exp", "log", "sqrt"}:
                    self.result_label.setText(f"Ошибка: переменная '{v}' не разрешена")
                    return
            # Проверяем, что есть решение
            if not (parent and hasattr(parent, 'last_sol') and parent.last_sol is not None):
                self.result_label.setText("Сначала решите задачу!")
                return
            def f(t):
                local_vars = {"t": t, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "sqrt": np.sqrt}
                xs = parent.last_sol(t)
                for i in range(n_vars):
                    local_vars[f"x{i+1}"] = xs[i]
                return eval(func_str, local_vars)
            val, err = quad(f, a, b)
            self.result_label.setText(f"{val:.8g}")
        except Exception as e:
            self.result_label.setText(f"Ошибка: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Решение краевой задачи")
        self.setMinimumSize(1200, 800)
        
        # Настройка стиля
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6f7;
            }
            QLabel {
                color: black;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QLineEdit, QSpinBox, QComboBox {
                padding: 7px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                color: black;
                font-size: 14px;
                min-height: 25px;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 2px solid #3498db;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                color: black;
                font-size: 14px;
            }
            QTableWidget::item {
                padding: 5px;
                color: black;
                font-size: 14px;
            }
            QTableWidget QLineEdit {
                font-size: 14px;
                color: black;
                padding: 2px;
                margin: 0px;
            }
            QHeaderView::section {
                background-color: #e1e1e1;
                padding: 7px;
                border: none;
                color: black;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #34495e;
                border-radius: 4px;
                margin-top: 10px;
                color: black;
                font-weight: bold;
                font-size: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: black;
                font-size: 15px;
            }
            QScrollArea {
                border: none;
            }
            QTextEdit {
                font-size: 14px;
                color: black;
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                text-align: center;
                min-height: 30px;
                margin: 5px 0px;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 3px;
            }
        """)
        
        self.init_ui()
        self.results = ResultsWindow(self)
    
    def init_ui(self):
        self.create_menu()
        central = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Левая колонка (уравнения и условия)
        left_column = QScrollArea()
        left_column.setWidgetResizable(True)
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setSpacing(15)
        left_column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_column.setMinimumWidth(800)  # Увеличиваем минимальную ширину для левой колонки

        # Добавляем поле для названия задачи
        name_frame = QFrame()
        name_layout = QHBoxLayout(name_frame)
        name_layout.addWidget(QLabel("Название задачи:"))
        self.task_name_input = QLineEdit()
        name_layout.addWidget(self.task_name_input)
        left_layout.addWidget(name_frame)

        # Группа для уравнений
        eq_group = QGroupBox("Система уравнений")
        eq_layout = QVBoxLayout()
        
        # Таблица уравнений
        self.equation_input = QTableWidget(1, 2)
        self.equation_input.setHorizontalHeaderLabels(["xi'", "Уравнение"])
        eq_layout.addWidget(self.equation_input)
        self.equation_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.equation_input.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.equation_input.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        
        eq_group.setLayout(eq_layout)
        left_layout.addWidget(eq_group)

        # Группа для краевых условий
        bc_group = QGroupBox("Краевые условия")
        bc_layout = QVBoxLayout()
        self.boundary_input = QTableWidget(1, 1)
        self.boundary_input.setHorizontalHeaderLabels(["Условие"])
        bc_layout.addWidget(self.boundary_input)
        self.boundary_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.boundary_input.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bc_group.setLayout(bc_layout)
        left_layout.addWidget(bc_group)

        # Группа для комментариев
        comments_group = QGroupBox("Комментарии")
        comments_layout = QVBoxLayout()
        self.comments_input = QTextEdit()
        self.comments_input.setPlaceholderText("Введите ваши заметки и комментарии здесь...")
        self.comments_input.setMinimumHeight(60)
        self.comments_input.setMaximumHeight(80)
        self.comments_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        comments_layout.addWidget(self.comments_input)
        comments_group.setLayout(comments_layout)
        left_layout.addWidget(comments_group)

        left_column.setWidget(left_content)
        main_layout.addWidget(left_column)

        # Правая колонка (параметры)
        right_column = QScrollArea()
        right_column.setWidgetResizable(True)
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setSpacing(15)
        right_column.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        right_column.setMaximumWidth(350)  # Уменьшаем максимальную ширину правой колонки

        # Группа для параметров
        param_group = QGroupBox("Параметры решения")
        param_layout = QVBoxLayout()
        
        # Создаем виджеты параметров
        self.num_eq_input = QLineEdit()
        self.a_input = QLineEdit()
        self.b_input = QLineEdit()
        self.t_star_input = QLineEdit()
        self.step_input = QLineEdit()
        self.iter_input = QLineEdit()
        self.tol_outer_input = QLineEdit()
        self.tol_inner_input = QLineEdit()
        self.grid_input = QLineEdit()
        self.p0_input = QLineEdit()
        self.nu_input = QLineEdit()
        self.method_outer = QComboBox()
        self.method_inner = QComboBox()
        
        self.method_outer.addItems(["euler", "runge-kutta", "Radau", "BDF"])
        self.method_inner.addItems(["euler", "runge-kutta", "Radau", "BDF"])
        
        # Создаем список параметров
        params = [
            ("Число уравнений", self.num_eq_input),
            ("Начало отрезка", self.a_input),
            ("Конец отрезка", self.b_input),
            ("Момент времени t*", self.t_star_input),
            ("Шаг вывода значений", self.step_input),
            ("Число итераций", self.iter_input),
            ("Точность внешней задачи", self.tol_outer_input),
            ("Точность внутренней задачи", self.tol_inner_input),
            ("Число узлов сетки", self.grid_input),
            ("p0", self.p0_input),
            ("nu", self.nu_input),
            ("Метод внешней задачи:", self.method_outer),
            ("Метод внутренней задачи:", self.method_inner)
        ]
        
        for label, widget in params:
            frame = QFrame()
            layout = QHBoxLayout(frame)
            layout.setSpacing(5)  # Уменьшаем расстояние между элементами
            label_widget = QLabel(label)
            label_widget.setMinimumWidth(120)  # Уменьшаем ширину меток
            layout.addWidget(label_widget)
            layout.addWidget(widget)
            param_layout.addWidget(frame)
        
        param_group.setLayout(param_layout)
        right_layout.addWidget(param_group)

        # Кнопки управления
        button_group = QGroupBox()
        button_layout = QHBoxLayout()
        self.solve_btn = QPushButton("Решить")
        self.solve_btn.clicked.connect(self.solve_task)
        button_layout.addWidget(self.solve_btn)
        button_group.setLayout(button_layout)
        right_layout.addWidget(button_group)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)

        right_column.setWidget(right_content)
        main_layout.addWidget(right_column)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _on_num_eq_changed(self):
        try:
            n = int(self.num_eq_input.text())
            if n < 1:
                n = 1
        except Exception:
            n = 1
        self.equation_input.setRowCount(n)
        self.boundary_input.setRowCount(n)
        self.p0_input.setText(", ".join(["0"] * n))
        for i in range(n):
            if not self.equation_input.item(i, 0):
                self.equation_input.setItem(i, 0, QTableWidgetItem(f"x{i + 1}'"))
            if not self.equation_input.item(i, 1):
                self.equation_input.setItem(i, 1, QTableWidgetItem(""))
            if not self.boundary_input.item(i, 0):
                self.boundary_input.setItem(i, 0, QTableWidgetItem(""))

    def create_menu(self):
        menubar = self.menuBar()

        # Меню "Файл"
        file_menu = menubar.addMenu("Файл")
        new_action = QAction("Новый", self)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        open_action = QAction("Открыть...", self)
        open_action.triggered.connect(self.load_task)
        file_menu.addAction(open_action)
        save_action = QAction("Сохранить", self)
        save_action.triggered.connect(self.save_task)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Меню "Библиотека примеров"
        examples_menu = menubar.addMenu("Библиотека примеров")
        example1_action = QAction("Краевая задача двух тел", self)
        example1_action.triggered.connect(self.load_example_1)
        examples_menu.addAction(example1_action)
        example2_action = QAction("Предельные циклы в системе Эквейлера", self)
        example2_action.triggered.connect(self.load_example_2)
        examples_menu.addAction(example2_action)
        example_integrator_nu_action = QAction("Задача для трехкратного интегратора", self)
        example_integrator_nu_action.triggered.connect(self.load_example_integrator_nu)
        examples_menu.addAction(example_integrator_nu_action)
        example_oscillator_action = QAction("Гармонический осциллятор", self)
        example_oscillator_action.triggered.connect(self.load_example_oscillator)
        examples_menu.addAction(example_oscillator_action)
        example_damped_pendulum_action = QAction("Маятник с затуханием", self)
        example_damped_pendulum_action.triggered.connect(self.load_example_damped_pendulum)
        examples_menu.addAction(example_damped_pendulum_action)
        example_quick_response_action = QAction("Задача быстродействия", self)
        example_quick_response_action.triggered.connect(self.load_example_quick_response)
        examples_menu.addAction(example_quick_response_action)
        # Можно добавить другие примеры аналогично

        # Меню "Помощь"
        help_menu = menubar.addMenu("Помощь")
        help_action = QAction("Руководство пользователя", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        syntax_action = QAction("Справка по синтаксису", self)
        syntax_action.triggered.connect(self.show_syntax_help)
        help_menu.addAction(syntax_action)

        # Меню "О программе"
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        menubar.addAction(about_action)

    def new_project(self):
        # Очищает все поля для нового проекта
        self.task_name_input.setText("")  # Очищаем название задачи
        self.comments_input.setText("")  # Очищаем комментарии
        self.num_eq_input.setText("1")
        self.a_input.setText("")
        self.b_input.setText("")
        self.iter_input.setText("")
        self.tol_outer_input.setText("")
        self.tol_inner_input.setText("")
        self.grid_input.setText("")
        self.p0_input.setText("")
        self.nu_input.setText("")
        self.method_outer.setCurrentIndex(0)
        self.method_inner.setCurrentIndex(0)
        self.progress_bar.setValue(0)  # Очищаем прогресс-бар
        self._on_num_eq_changed()
        for i in range(self.equation_input.rowCount()):
            self.equation_input.setItem(i, 1, QTableWidgetItem(""))
        for i in range(self.boundary_input.rowCount()):
            self.boundary_input.setItem(i, 0, QTableWidgetItem(""))
        if hasattr(self, 'basis_group'):
            self.basis_group.hide()

    def show_syntax_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Справка по синтаксису")
        dialog.setMinimumSize(900, 700)
        layout = QVBoxLayout()
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>Справка по синтаксису уравнений</h2>

        <h3>1. Основные математические операции</h3>
        <ul>
            <li><b>Сложение:</b> x1 + x2</li>
            <li><b>Вычитание:</b> x1 - x2</li>
            <li><b>Умножение:</b> x1 * x2</li>
            <li><b>Деление:</b> x1 / x2</li>
            <li><b>Возведение в степень:</b> x1^2 или x1**2</li>
            <li><b>Группировка:</b> (x1 + x2) * x3</li>
        </ul>

        <h3>2. Математические функции</h3>
        <ul>
            <li><b>Тригонометрические:</b>
                <ul>
                    <li>sin(x) - синус</li>
                    <li>cos(x) - косинус</li>
                    <li>tan(x) - тангенс</li>
                    <li>arcsin(x) - арксинус</li>
                    <li>arccos(x) - арккосинус</li>
                    <li>arctan(x) - арктангенс</li>
                </ul>
            </li>
            <li><b>Экспоненциальные и логарифмические:</b>
                <ul>
                    <li>exp(x) - экспонента (e^x)</li>
                    <li>log(x) - натуральный логарифм</li>
                    <li>log10(x) - десятичный логарифм</li>
                </ul>
            </li>
            <li><b>Степенные и корни:</b>
                <ul>
                    <li>sqrt(x) - квадратный корень</li>
                    <li>x^2 - квадрат</li>
                    <li>x^3 - куб</li>
                </ul>
            </li>
            <li><b>Гиперболические:</b>
                <ul>
                    <li>sinh(x) - гиперболический синус</li>
                    <li>cosh(x) - гиперболический косинус</li>
                    <li>tanh(x) - гиперболический тангенс</li>
                </ul>
            </li>
        </ul>

        <h3>3. Специальные функции</h3>
        <ul>
            <li><b>Абсолютное значение:</b> abs(x)</li>
            <li><b>Знак числа:</b> sign(x)</li>
            <li><b>Целая часть:</b> floor(x)</li>
            <li><b>Округление:</b> round(x)</li>
            <li><b>Минимум/максимум:</b> min(x1, x2), max(x1, x2)</li>
        </ul>

        <h3>4. Константы</h3>
        <ul>
            <li><b>π (пи):</b> pi</li>
            <li><b>e (основание натурального логарифма):</b> e</li>
        </ul>

        <h3>5. Примеры уравнений</h3>
        <p><b>Простая система:</b></p>
        <pre>
x1' = x2
x2' = -x1
        </pre>

        <p><b>Система с параметрами:</b></p>
        <pre>
x1' = a*x1 + b*x2
x2' = c*x1 + d*x2
        </pre>

        <p><b>Нелинейная система:</b></p>
        <pre>
x1' = x2
x2' = -0.1*x2 - x1^3 + cos(t)
        </pre>

        <p><b>Система с тригонометрическими функциями:</b></p>
        <pre>
x1' = sin(x2)
x2' = cos(x1)
        </pre>

        <h3>6. Правила записи</h3>
        <ul>
            <li>Используйте x1, x2, x3, ... для обозначения переменных</li>
            <li>Время обозначается как t</li>
            <li>Все математические операции должны быть явными (например, 2*x, а не 2x)</li>
            <li>Используйте скобки для группировки выражений</li>
            <li>Для возведения в степень используйте ^ или **</li>
            <li>Аргументы функций заключаются в скобки: sin(x), cos(t)</li>
        </ul>

        <h3>7. Частые ошибки</h3>
        <ul>
            <li>Пропуск знака умножения: 2x вместо 2*x</li>
            <li>Неправильное использование скобок</li>
            <li>Использование неопределенных переменных</li>
            <li>Неправильный порядок операций</li>
            <li>Использование недопустимых символов</li>
        </ul>

        <h3>8. Советы по вводу</h3>
        <ul>
            <li>Проверяйте правильность расстановки скобок</li>
            <li>Используйте пробелы для лучшей читаемости</li>
            <li>Разбивайте сложные выражения на части</li>
            <li>Проверяйте размерность уравнений</li>
            <li>Убедитесь, что все переменные определены</li>
        </ul>
        """)
        
        layout.addWidget(help_text)
        
        # Кнопка закрытия
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def show_about(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("О программе")
        layout = QVBoxLayout()
        
        # Добавляем фотографию автора
        photo_label = QLabel()
        try:
            pixmap = QPixmap("author_photo.jpg")
            if not pixmap.isNull():
                pixmap = pixmap.scaledToWidth(180, Qt.SmoothTransformation)
                photo_label.setPixmap(pixmap)
                photo_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(photo_label)
        except Exception as e:
            pass  # Если не удалось загрузить фото, просто не показываем
        
        text = QLabel("""
        Решатель краевых задач
        
        Версия 1.0
        
        Программа предназначена для численного решения 
        систем обыкновенных дифференциальных уравнений 
        с краевыми условиями.
        
        Московский государственный университет имени М.В. Ломоносова
        Факультет вычислительной математики и кибернетики
        Кафедра оптимального управления
        Преподаватель: Аввакумов Сергей Николаевич
        
        Разработчик: Царьков Денис
        Год: 2025
        """)
        layout.addWidget(text)
        btn = QPushButton("ОК")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        dialog.setLayout(layout)
        dialog.exec_()

    def save_task(self):
        # Сначала сохраняем условия в txt файл
        txt_filename, _ = QFileDialog.getSaveFileName(self, "Сохранить условия задачи", "", "Text Files (*.txt)")
        if txt_filename:
            with open(txt_filename, 'w', encoding='utf-8') as f:
                # Сохраняем название задачи
                f.write(f"Название задачи: {self.task_name_input.text()}\n\n")
                # Сохраняем комментарии
                f.write(f"Комментарии:\n{self.comments_input.toPlainText()}\n\n")
                # Сохраняем уравнения
                f.write("Система уравнений:\n")
                for i in range(self.equation_input.rowCount()):
                    left = self.equation_input.item(i, 0).text() if self.equation_input.item(i, 0) else ""
                    right = self.equation_input.item(i, 1).text() if self.equation_input.item(i, 1) else ""
                    f.write(f"{left} {right}\n")
                f.write("\nКраевые условия:\n")
                for i in range(self.boundary_input.rowCount()):
                    cond = self.boundary_input.item(i, 0).text() if self.boundary_input.item(i, 0) else ""
                    f.write(f"{cond}\n")
                f.write("\nПараметры:\n")
                f.write(f"Число уравнений: {self.num_eq_input.text()}\n")
                f.write(f"Начало отрезка: {self.a_input.text()}\n")
                f.write(f"Конец отрезка: {self.b_input.text()}\n")
                f.write(f"Число итераций: {self.iter_input.text()}\n")
                f.write(f"Точность внешней задачи: {self.tol_outer_input.text()}\n")
                f.write(f"Точность внутренней задачи: {self.tol_inner_input.text()}\n")
                f.write(f"Число узлов сетки: {self.grid_input.text()}\n")
                f.write(f"p0: {self.p0_input.text()}\n")
                f.write(f"nu: {self.nu_input.text()}\n")
                f.write(f"Метод внешней задачи: {self.method_outer.currentText()}\n")
                f.write(f"Метод внутренней задачи: {self.method_inner.currentText()}\n")

            # Если есть результаты, сохраняем их в Excel
            if hasattr(self.results, "last_x") and hasattr(self.results, "last_y"):
                try:
                    excel_filename, _ = QFileDialog.getSaveFileName(self, "Сохранить результаты", "", "Excel Files (*.xlsx)")
                    if excel_filename:
                        # Проверяем и корректируем размерности перед созданием DataFrame
                        t_values = self.results.last_x
                        y_values = self.results.last_y

                        if t_values.shape[0] != y_values.shape[0]:
                            print(f"Warning: time array length ({t_values.shape[0]}) != solution array rows ({y_values.shape[0]})")
                            min_len = min(t_values.shape[0], y_values.shape[0])
                            t_values = t_values[:min_len]
                            y_values = y_values[:min_len, :]

                        # Создаем DataFrame с результатами
                        data = {'t': t_values}
                        for i in range(y_values.shape[1]):
                            data[f'x{i+1}'] = y_values[:, i]
                        df = pd.DataFrame(data)

                        # Сохраняем в Excel
                        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                            # Сохраняем решение
                            df.to_excel(writer, sheet_name='Решение', index=False)
                            
                            # Добавляем лист с информацией о задаче
                            info_data = {
                                'Параметр': [
                                    'Название задачи',
                                    'Число уравнений',
                                    'Начало отрезка',
                                    'Конец отрезка',
                                    'Число итераций',
                                    'Точность внешней задачи',
                                    'Точность внутренней задачи',
                                    'Число узлов сетки',
                                    'p0',
                                    'nu',
                                    'Метод внешней задачи',
                                    'Метод внутренней задачи'
                                ],
                                'Значение': [
                                    self.task_name_input.text(),
                                    self.num_eq_input.text(),
                                    self.a_input.text(),
                                    self.b_input.text(),
                                    self.iter_input.text(),
                                    self.tol_outer_input.text(),
                                    self.tol_inner_input.text(),
                                    self.grid_input.text(),
                                    self.p0_input.text(),
                                    self.nu_input.text(),
                                    self.method_outer.currentText(),
                                    self.method_inner.currentText()
                                ]
                            }
                            pd.DataFrame(info_data).to_excel(writer, sheet_name='Параметры', index=False)
                            
                            # Добавляем лист с уравнениями
                            eq_data = {
                                'Уравнение': [f"{self.equation_input.item(i, 0).text()} = {self.equation_input.item(i, 1).text()}" 
                                            for i in range(self.equation_input.rowCount())]
                            }
                            pd.DataFrame(eq_data).to_excel(writer, sheet_name='Уравнения', index=False)
                            
                            # Добавляем лист с краевыми условиями
                            bc_data = {
                                'Краевое условие': [self.boundary_input.item(i, 0).text() 
                                                  for i in range(self.boundary_input.rowCount())]
                            }
                            pd.DataFrame(bc_data).to_excel(writer, sheet_name='Краевые условия', index=False)
                            
                        # Показываем сообщение об успешном сохранении
                        from PySide6.QtWidgets import QMessageBox
                        QMessageBox.information(self, "Успех", "Результаты успешно сохранены в Excel файл")
                except Exception as e:
                    # Показываем сообщение об ошибке
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результаты в Excel: {str(e)}")

    def load_task(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Загрузить задачу", "", "Text Files (*.txt)")
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            section = None
            eqs = []
            bcs = []
            params = {}
            task_name = ""
            comments = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Название задачи:"):
                    task_name = line.replace("Название задачи:", "").strip()
                    continue
                elif line.startswith("Комментарии:"):
                    section = "comments"
                    continue
                elif line.startswith("Система уравнений"):
                    section = "eq"
                    continue
                elif line.startswith("Краевые условия"):
                    section = "bc"
                    continue
                elif line.startswith("Параметры"):
                    section = "params"
                    continue
                
                if section == "comments":
                    comments += line + "\n"
                elif section == "eq":
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        eqs.append((parts[0], parts[1]))
                elif section == "bc":
                    bcs.append(line)
                elif section == "params":
                    if ':' in line:
                        k, v = line.split(':', 1)
                        params[k.strip()] = v.strip()
            # Установить название задачи
            self.task_name_input.setText(task_name)
            # Установить комментарии
            self.comments_input.setText(comments.strip())
            # Установить число уравнений
            n = int(params.get("Число уравнений", len(eqs)))
            self.num_eq_input.setText(str(n))
            self.equation_input.setRowCount(n)
            self.boundary_input.setRowCount(n)
            # Заполнить уравнения
            for i in range(n):
                left = f"x{i+1}'"
                right = ""
                if i < len(eqs):
                    left, right = eqs[i]
                self.equation_input.setItem(i, 0, QTableWidgetItem(left))
                self.equation_input.setItem(i, 1, QTableWidgetItem(right))
            # Заполнить краевые условия
            for i in range(n):
                val = bcs[i] if i < len(bcs) else ""
                self.boundary_input.setItem(i, 0, QTableWidgetItem(val))
            # Параметры
            self.a_input.setText(params.get("Начало отрезка", ""))
            self.b_input.setText(params.get("Конец отрезка", ""))
            self.iter_input.setText(params.get("Число итераций", ""))
            self.tol_outer_input.setText(params.get("Точность внешней задачи", ""))
            self.tol_inner_input.setText(params.get("Точность внутренней задачи", ""))
            self.grid_input.setText(params.get("Число узлов сетки", ""))
            self.p0_input.setText(params.get("p0", ""))
            self.nu_input.setText(params.get("nu", ""))
            outer = params.get("Метод внешней задачи", "euler")
            inner = params.get("Метод внутренней задачи", "euler")
            self.method_outer.setCurrentText(outer)
            self.method_inner.setCurrentText(inner)

    def solve_task(self):
        try:
            # Проверяем, является ли текущая задача задачей быстродействия
            if self.task_name_input.text() == "Задача быстродействия":
                self.show_quick_response_plots()
                return

            # Сбор данных из GUI
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            t_span = (a, b)
            mu_span = (0, 1)
    
            # Параметры от пользователя
            tol_inner = float(self.tol_inner_input.text())
            tol_corrector = float(self.tol_outer_input.text())
            N_mesh = int(self.grid_input.text())
            max_steps = int(self.iter_input.text())
            method_inner = 'RK45' if self.method_inner.currentText() == 'runge-kutta' else 'RK23'
    
            # Начальные условия
            p0_text = self.p0_input.text()
            p0 = np.array([float(val.strip()) for val in p0_text.split(',')])
    
            # Уравнения
            equations = []
            for i in range(self.equation_input.rowCount()):
                item = self.equation_input.item(i, 1)
                if item:
                    eq = item.text().strip()
                    if eq:
                        equations.append(eq)
    
            # Граничные условия
            bc_list = []
            for i in range(self.boundary_input.rowCount()):
                item = self.boundary_input.item(i, 0)
                if item:
                    cond = item.text().strip()
                    if cond:
                        bc_list.append(cond)
    
            # Сброс прогресс-бара
            self.progress_bar.setValue(0)
    
            # Решение методом продолжения
            p_final, sol_func, t_eval_output, x_at_grid_points = continuation_method(
                p0, t_span, mu_span,
                tol_inner=tol_inner,
                tol_corrector=tol_corrector,
                method_inner=method_inner,
                f_system_equations=equations,
                bc_list=bc_list,
                corrector_method='hybr',
                n_mu_steps=100,
                max_corrector_steps=max_steps,
                N_mesh_inner=N_mesh,
                max_step_inner=None,
                max_steps_inner=None,
                grid_points_for_output=500,
                progress_callback=self.progress_bar.setValue
            )
    
            # Сохраняем результаты для отображения
            self.results.last_x = t_eval_output
            self.results.last_y = x_at_grid_points.T  # Транспонируем для правильной формы (n_vars, n_points)
            print(f"Solution shapes - t_eval: {t_eval_output.shape}, x_at_grid: {x_at_grid_points.shape}")
            
            # Обновляем селекторы осей
            self.results.update_axis_selectors(len(p0))

            # Отображаем результаты
            self.results.set_task_title(self.task_name_input.text())
            self.results.display_results(t_eval_output, x_at_grid_points.T, sol_func, t_span)
            self.progress_bar.setValue(100)
            self.results.exec_()

        except Exception as e:
            print(f"Error in solve_task: {str(e)}")
            self.results.result_table.setText(f"Ошибка при решении задачи:\n{e}")
            self.progress_bar.setValue(0)
            self.results.set_task_title("") # Сбросить заголовок при ошибке
            self.results.exec_()
    
    def load_example_1(self):
        self.progress_bar.setValue(0)  # Очищаем прогресс-бар
        self.task_name_input.setText("Краевая задача двух тел")
        self.comments_input.setText("Пример краевой задачи для системы двух тел, взаимодействующих по закону всемирного тяготения.")
        n = 4
        self.num_eq_input.setText(str(n))
        self.equation_input.setRowCount(n)
        self.boundary_input.setRowCount(n)
        # Заполнить первый столбец xi'
        for i in range(n):
            self.equation_input.setItem(i, 0, QTableWidgetItem(f"x{i + 1}'"))
            self.equation_input.setItem(i, 1, QTableWidgetItem(""))  # очистить
            self.boundary_input.setItem(i, 0, QTableWidgetItem(""))  # очистить

        eqs = [
            "x3",
            "x4",
            "-x1 * (x1**2 + x2**2)**(-3/2)",
            "-x2 * (x1**2 + x2**2)**(-3/2)"
        ]
        for i, eq in enumerate(eqs):
            self.equation_input.setItem(i, 1, QTableWidgetItem(eq))
        bcs = [
            "x1(0)=2",
            "x2(0)=0",
            "x1(7)=1.0738644361",
            "x2(7)=-1.0995343576"
        ]
        for i, bc in enumerate(bcs):
            self.boundary_input.setItem(i, 0, QTableWidgetItem(bc))
        # Остальные параметры
        self.a_input.setText("0")
        self.b_input.setText("7")
        self.iter_input.setText("10000")
        self.tol_outer_input.setText("1e-6")
        self.tol_inner_input.setText("1e-6")
        self.grid_input.setText("100")
        self.p0_input.setText("2, 0, -0.5, 0.5")
        self.nu_input.setText("1e-6")
        self.method_inner.setCurrentText("runge-kutta")
        self.method_outer.setCurrentText("runge-kutta")
        if hasattr(self, 'basis_group'):
            self.basis_group.hide()

    def load_example_2(self):
        self.progress_bar.setValue(0)  # Очищаем прогресс-бар
        self.task_name_input.setText("Предельные циклы в системе Эквейлера")
        self.comments_input.setText("Пример задачи на поиск предельных циклов в системе Эквейлера.")
        import numpy as np
        n = 4
        self.num_eq_input.setText(str(n))
        self.equation_input.setRowCount(n)
        self.boundary_input.setRowCount(n)
        # Заполнить первый столбец xi'
        for i in range(n):
            self.equation_input.setItem(i, 0, QTableWidgetItem(f"x{i + 1}'"))
            self.equation_input.setItem(i, 1, QTableWidgetItem(""))
            self.boundary_input.setItem(i, 0, QTableWidgetItem(""))

        eqs = [
            "x3*x2",
            "x3*(-x1 + np.sin(x2))",
            "0",
            "0"
        ]
        for i, eq in enumerate(eqs):
            self.equation_input.setItem(i, 1, QTableWidgetItem(eq))
        bcs = [
            "x1(0)=x4(0)",
            "x2(0)=0",
            "x1(1)=x4(1)",
            "x2(1)=0"
        ]
        for i, bc in enumerate(bcs):
            self.boundary_input.setItem(i, 0, QTableWidgetItem(bc))
        # Остальные параметры
        self.a_input.setText("0")
        self.b_input.setText("1")
        self.iter_input.setText("10000")
        self.tol_outer_input.setText("1e-6")
        self.tol_inner_input.setText("1e-6")
        self.grid_input.setText("100")
        self.p0_input.setText(f"2, 0, {2*np.pi}, 2")
        self.nu_input.setText("1e-6")
        self.method_inner.setCurrentText("runge-kutta")
        self.method_outer.setCurrentText("runge-kutta")
        if hasattr(self, 'basis_group'):
            self.basis_group.hide()

    def load_example_integrator_nu(self):
        self.progress_bar.setValue(0)  # Очищаем прогресс-бар
        self.task_name_input.setText("Краевая задача с малым параметром")
        self.comments_input.setText("Пример краевой задачи для трехкратного интегратора с малым параметром.")
        n = 6
        self.num_eq_input.setText(str(n))
        self.equation_input.setRowCount(n)
        self.boundary_input.setRowCount(n)
        nu = float(self.nu_input.text()) if self.nu_input.text() else 1e-10
        T = 3.275
        self.nu_input.setText(str(nu))
        # Заполнить xi'
        for i in range(n):
            self.equation_input.setItem(i, 0, QTableWidgetItem(f"x{i + 1}'"))
            self.equation_input.setItem(i, 1, QTableWidgetItem(""))
            self.boundary_input.setItem(i, 0, QTableWidgetItem(""))
        eqs = [
            "x2",
            "x3",
            f"0.5 * (np.sqrt({nu} + (x6 + 1)**2) - np.sqrt({nu} + (x6 - 1)**2))",
            "0",
            "-x4",
            "-x5"
        ]
        for i, eq in enumerate(eqs):
            self.equation_input.setItem(i, 1, QTableWidgetItem(eq))
        bcs = [
            "x1(0)=1",
            "x2(0)=0",
            "x3(0)=0",
            f"x1({T})=0",
            f"x2({T})=0",
            f"x3({T})=0"
        ]
        for i, bc in enumerate(bcs):
            self.boundary_input.setItem(i, 0, QTableWidgetItem(bc))
        # Остальные параметры
        self.a_input.setText("0")
        self.b_input.setText(str(T))
        self.iter_input.setText("2000")
        self.tol_outer_input.setText("1e-4")
        self.tol_inner_input.setText("1e-4")
        self.grid_input.setText("100")
        self.p0_input.setText("1, 0, 0, 0, 0, 0")
        self.method_inner.setCurrentText("Radau")
        self.method_outer.setCurrentText("Radau")
        if hasattr(self, 'basis_group'):
            self.basis_group.hide()

    def load_example_oscillator(self):
        self.progress_bar.setValue(0)
        self.task_name_input.setText("Гармонический осциллятор")
        self.comments_input.setText("Классический гармонический осциллятор: x'' + w^2 x = 0")
        n = 2
        self.num_eq_input.setText(str(n))
        self.equation_input.setRowCount(n)
        self.boundary_input.setRowCount(n)
        w = 2.0
        eqs = [
            "x2",
            f"-{w}**2*x1"
        ]
        for i in range(n):
            self.equation_input.setItem(i, 0, QTableWidgetItem(f"x{i+1}'"))
            self.equation_input.setItem(i, 1, QTableWidgetItem(eqs[i]))
        bcs = [
            "x1(0)=1",
            "x2(0)=0"
        ]
        for i in range(n):
            self.boundary_input.setItem(i, 0, QTableWidgetItem(bcs[i] if i < len(bcs) else ""))
        self.a_input.setText("0")
        self.b_input.setText("10")
        self.tol_outer_input.setText("1e-6")
        self.tol_inner_input.setText("1e-6")
        self.grid_input.setText("200")
        self.p0_input.setText("1, 0")
        self.nu_input.setText("")
        self.method_inner.setCurrentText("runge-kutta")
        self.method_outer.setCurrentText("runge-kutta")
        self.iter_input.setText("10000")
        if hasattr(self, 'basis_group'):
            self.basis_group.hide()

    def load_example_damped_pendulum(self):
        self.progress_bar.setValue(0)
        self.task_name_input.setText("Маятник с затуханием")
        self.comments_input.setText("Линейный маятник с затуханием: x'' + 2b x' + w^2 x = 0")
        n = 2
        self.num_eq_input.setText(str(n))
        self.equation_input.setRowCount(n)
        self.boundary_input.setRowCount(n)
        w = 1.5
        b = 0.3
        eqs = [
            "x2",
            f"-2*{b}*x2 - {w}**2*x1"
        ]
        for i in range(n):
            self.equation_input.setItem(i, 0, QTableWidgetItem(f"x{i+1}'"))
            self.equation_input.setItem(i, 1, QTableWidgetItem(eqs[i]))
        bcs = [
            "x1(0)=1",
            "x2(0)=0"
        ]
        for i in range(n):
            self.boundary_input.setItem(i, 0, QTableWidgetItem(bcs[i] if i < len(bcs) else ""))
        self.a_input.setText("0")
        self.b_input.setText("10")
        self.tol_outer_input.setText("1e-6")
        self.tol_inner_input.setText("1e-6")
        self.grid_input.setText("200")
        self.p0_input.setText("1, 0")
        self.nu_input.setText("")
        self.method_inner.setCurrentText("runge-kutta")
        self.method_outer.setCurrentText("runge-kutta")
        self.iter_input.setText("10000")
        if hasattr(self, 'basis_group'):
            self.basis_group.hide()

    def load_example_quick_response(self):
        self.progress_bar.setValue(0)
        self.task_name_input.setText("Задача быстродействия")
        self.comments_input.setText("Пример задачи быстродействия с предварительно заданными управлениями.")
        n = 4
        self.num_eq_input.setText(str(n))
        self.equation_input.setRowCount(n)
        self.boundary_input.setRowCount(n)
        
        # Заполняем уравнения
        eqs = [
            "x2 + c_phi1",
            "x1 * -1.532 - 0.6323 * x2 + c_phi2",
            "x4",
            "u1"
        ]
        for i in range(n):
            self.equation_input.setItem(i, 0, QTableWidgetItem(f"x{i+1}'"))
            self.equation_input.setItem(i, 1, QTableWidgetItem(eqs[i]))
        
        # Заполняем краевые условия
        bcs = [
            "x1(0)=0",
            "x2(0)=0",
            "x3(0)=0",
            "x4(0)=0",
            "x1(1)=1",
            "x2(1)=0"
        ]
        for i in range(n):
            self.boundary_input.setItem(i, 0, QTableWidgetItem(bcs[i] if i < len(bcs) else ""))
        
        # Устанавливаем параметры
        self.a_input.setText("0")
        self.b_input.setText("1")
        self.tol_outer_input.setText("1e-6")
        self.tol_inner_input.setText("1e-6")
        self.grid_input.setText("500")
        self.p0_input.setText("0, 0, 0, 0")
        self.nu_input.setText("")
        self.method_inner.setCurrentText("runge-kutta")
        self.method_outer.setCurrentText("runge-kutta")
        self.iter_input.setText("1000")

    def show_quick_response_plots(self):
        import numpy as np
        
        # Создаем данные для графиков
        t = np.linspace(0, 1, 500)
        
        # Данные для u1
        t_a, t_b = 0.0, 0.15
        t_c, t_d = 0.45, 0.55
        u1 = np.piecewise(t,
            [(t > t_a) & (t < t_b),
             (t >= t_b) & (t < t_c),
             (t >= t_c) & (t < t_d),
             t >= t_d],
            [lambda t: -0.5 * np.sin(np.pi * (t - t_a) / (t_b - t_a)),
             0,
             lambda t: 0.5 * np.sin(np.pi * (t - t_c) / (t_d - t_c)),
             0])
        
        # Данные для u2
        t1, t2, t3, t4 = 0.2, 0.5, 0.8, 1.0
        u2 = np.piecewise(t,
            [t < t1,
             (t >= t1) & (t < t2),
             (t >= t2) & (t < t3),
             t >= t3],
            [lambda t: (t - 0)/(t1 - 0)*1,
             1,
             lambda t: 1 + (t - t2)/(t3 - t2)*(-2),
             -1])
        
        # Создаем массив для отображения в ResultsWindow
        y = np.vstack((u1, u2))
        
        # Создаем функцию-обертку для решения
        def sol_func(t):
            if isinstance(t, np.ndarray):
                return np.array([np.array([u1[i], u2[i]]) for i in range(len(t))])
            idx = np.abs(t - t).argmin()
            return np.array([u1[idx], u2[idx]])
        
        # Обновляем окно результатов
        self.results.update_axis_selectors(2)  # 2 переменные управления
        self.results.set_task_title(self.task_name_input.text())
        self.results.display_results(t, y, sol_func, (0, 1))
        
        # Настраиваем график
        self.results.plot.ax.clear()
        self.results.plot.ax.plot(t, u1, '-', linewidth=2, label='$u_1(t)$')
        self.results.plot.ax.plot(t, u2, '--', linewidth=2, label='$u_2(t)$')
        
        # Добавляем маркеры
        self.results.plot.ax.plot(0, 0, marker='o', markerfacecolor='white', markeredgecolor='black')
        self.results.plot.ax.plot(t_b, 0, marker='o', markerfacecolor='white', markeredgecolor='black')
        self.results.plot.ax.plot(t_c, 0, marker='o', markerfacecolor='white', markeredgecolor='black')
        self.results.plot.ax.plot(t_d, 0, marker='o', markerfacecolor='white', markeredgecolor='black')
        self.results.plot.ax.plot(1, -1, marker='o', markerfacecolor='white', markeredgecolor='black')
        
        # Настраиваем оси и сетку
        self.results.plot.ax.set_ylim(-1.5, 1.5)
        self.results.plot.ax.set_xlim(-0.05, 1.05)
        self.results.plot.ax.set_xlabel('$t$')
        self.results.plot.ax.set_ylabel('$u(t)$')
        self.results.plot.ax.grid(True)
        self.results.plot.ax.legend()
        
        # Обновляем отображение
        self.results.plot.draw()
        
        # Показываем окно результатов
        self.results.exec_()

    def show_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Помощь")
        dialog.setMinimumSize(800, 600)
        layout = QVBoxLayout()
        
        # Создаем текстовое поле с прокруткой
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>Руководство пользователя</h2>
        
        <h3>1. Ввод системы уравнений</h3>
        <p>В таблице "Система уравнений" введите правые части дифференциальных уравнений:</p>
        <ul>
            <li>Используйте x1, x2, x3, ... для обозначения переменных</li>
            <li>Доступны математические функции: sin(), cos(), exp(), log(), sqrt()</li>
            <li>Пример: x2' = -0.1*x2 - x1^3 + cos(t)</li>
        </ul>

        <h3>2. Краевые условия</h3>
        <p>В таблице "Краевые условия" укажите условия на концах отрезка:</p>
        <ul>
            <li>Формат: xi(a)=value или xi(b)=value</li>
            <li>Пример: x1(0)=1, x2(1)=0</li>
            <li>Можно задавать равенство между переменными: x1(0)=x2(0)</li>
        </ul>

        <h3>3. Параметры решения</h3>
        <ul>
            <li><b>Число уравнений:</b> количество уравнений в системе</li>
            <li><b>Начало/конец отрезка:</b> границы временного интервала [a,b]</li>
            <li><b>Момент времени t*:</b> точка, в которой нужно вывести решение</li>
            <li><b>Шаг вывода значений:</b> интервал между точками в таблице результатов</li>
            <li><b>Число итераций:</b> максимальное количество шагов метода</li>
            <li><b>Точность внешней/внутренней задачи:</b> допустимая погрешность</li>
            <li><b>Число узлов сетки:</b> количество точек для численного решения</li>
            <li><b>p0:</b> начальное приближение (через запятую)</li>
            <li><b>nu:</b> малый параметр (если требуется)</li>
        </ul>

        <h3>4. Методы решения</h3>
        <p>Доступные методы для внутренней и внешней задачи:</p>
        <ul>
            <li><b>euler:</b> метод Эйлера (простой, но менее точный)</li>
            <li><b>runge-kutta:</b> метод Рунге-Кутты (более точный)</li>
            <li><b>Radau:</b> метод Радо (хорош для жестких систем)</li>
            <li><b>BDF:</b> метод обратного дифференцирования (для жестких систем)</li>
        </ul>

        <h3>5. Работа с результатами</h3>
        <p>После решения задачи вы увидите окно с результатами:</p>
        <ul>
            <li>График решения с возможностью выбора осей</li>
            <li>Таблица значений в заданных точках</li>
            <li>Возможность построения графиков произвольных выражений</li>
            <li>Вычисление интегралов от функций решения</li>
        </ul>

        <h3>6. Сохранение и загрузка</h3>
        <p>Программа позволяет:</p>
        <ul>
            <li>Сохранять условия задачи в текстовый файл</li>
            <li>Сохранять результаты в Excel</li>
            <li>Загружать сохраненные задачи</li>
            <li>Использовать готовые примеры из библиотеки</li>
        </ul>

        <h3>7. Возможные ошибки</h3>
        <p>Если возникают проблемы:</p>
        <ul>
            <li>Проверьте правильность ввода уравнений и условий</li>
            <li>Убедитесь, что число уравнений совпадает с числом условий</li>
            <li>Попробуйте изменить начальное приближение p0</li>
            <li>Увеличьте точность или число итераций</li>
            <li>Попробуйте другой метод решения</li>
        </ul>

        <h3>8. Примеры задач</h3>
        <p>В меню "Библиотека примеров" доступны:</p>
        <ul>
            <li>Краевая задача двух тел</li>
            <li>Предельные циклы в системе Эквейлера</li>
            <li>Задача для трехкратного интегратора</li>
            <li>Гармонический осциллятор</li>
            <li>Маятник с затуханием</li>
            <li>Задача быстродействия</li>
        </ul>
        """)
        
        layout.addWidget(help_text)
        
        # Кнопка закрытия
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
