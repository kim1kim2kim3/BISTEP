"""
==============================================================================
해양 환경에서 희생 양극(Sacrificial Anode)의 갈바닉 부식 시뮬레이션
==============================================================================

이 코드는 유한 차분법(FDM)을 사용하여 2D 도메인에서 갈바닉 부식 과정을 시뮬레이션합니다.

물리적 배경:
- 희생 양극(아연, 마그네슘 등)이 선체(강철)를 보호하기 위해 사용됨
- 아노드는 자발적으로 부식되어 전자를 선체에 공급
- 이 과정에서 아노드의 형상이 점진적으로 줄어듦

주요 방정식:
1. 라플라스 방정식: ∇²φ = 0 (정상 상태 전위 분포)
2. 전류 밀도: j = -σ∇φ (옴의 법칙)
3. 패러데이 법칙: 부식 속도 ∝ 전류 밀도

저자: AI Assistant
날짜: 2025
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 물리 상수 및 시뮬레이션 파라미터 설정
# ==============================================================================

class SimulationParameters:
    """시뮬레이션에 사용되는 모든 물리적 파라미터를 관리하는 클래스"""
    
    def __init__(self):
        # ----- 그리드 파라미터 -----
        self.nx = 60  # x 방향 그리드 수
        self.ny = 60  # y 방향 그리드 수
        self.dx = 0.01  # 그리드 간격 [m] (1cm)
        self.dy = 0.01  # 그리드 간격 [m]
        
        # ----- 물리적 파라미터 -----
        self.sigma = 4.5  # 해수 전도도 [S/m] (4~5 S/m 범위)
        self.phi_anode = -1.0  # 아노드(희생 양극) 전위 [V]
        self.phi_cathode = -0.6  # 캐소드(선체) 전위 [V]
        
        # ----- 아노드 재료 파라미터 (아연 기준) -----
        self.M_anode = 65.38e-3  # 아연 몰질량 [kg/mol]
        self.z_anode = 2  # 아연 산화 반응의 전자 수 (Zn → Zn²⁺ + 2e⁻)
        self.rho_anode = 7140  # 아연 밀도 [kg/m³]
        self.F = 96485  # 패러데이 상수 [C/mol]
        
        # ----- 시뮬레이션 파라미터 -----
        self.n_timesteps = 200  # 총 시간 스텝 수
        self.dt_real = 3600  # 실제 시간 스텝 [s] (1시간)
        self.acceleration_factor = 500  # 시간 가속 계수 (시뮬레이션 속도 향상)
        self.dt = self.dt_real * self.acceleration_factor  # 가속된 시간 스텝
        
        # ----- 수치 해석 파라미터 -----
        self.max_iter = 5000  # 라플라스 방정식 풀이 최대 반복 횟수
        self.tolerance = 1e-6  # 수렴 판정 오차
        self.omega = 1.7  # SOR(Successive Over-Relaxation) 파라미터
        
        # ----- 부식 임계값 -----
        self.erosion_threshold = 1.0  # 픽셀 제거 임계값 (0~1)

# ==============================================================================
# 2. 도메인 및 형상 초기화
# ==============================================================================

def initialize_domain(params):
    """
    시뮬레이션 도메인을 초기화합니다.
    
    도메인 구성:
    - 0: 해수(Electrolyte) - 전해질 영역
    - 1: 아노드(Sacrificial Anode) - 희생 양극
    - 2: 캐소드(Cathode/Hull) - 선체
    
    Returns:
        domain: 도메인 배열 (0: 해수, 1: 아노드, 2: 캐소드)
        phi: 초기 전위 분포
        erosion_state: 각 아노드 픽셀의 부식 진행 상태 (0~1)
    """
    domain = np.zeros((params.ny, params.nx), dtype=int)
    
    # ----- 아노드 배치 (왼쪽 중앙에 직사각형) -----
    # 실제 환경에서 희생 양극은 선체에 부착됨
    anode_x_start, anode_x_end = 5, 15
    anode_y_start, anode_y_end = 20, 40
    domain[anode_y_start:anode_y_end, anode_x_start:anode_x_end] = 1
    
    # ----- 캐소드(선체) 배치 (오른쪽에 수직 벽면) -----
    cathode_x_start, cathode_x_end = params.nx - 8, params.nx - 3
    cathode_y_start, cathode_y_end = 10, 50
    domain[cathode_y_start:cathode_y_end, cathode_x_start:cathode_x_end] = 2
    
    # ----- 초기 전위 분포 설정 -----
    phi = np.zeros((params.ny, params.nx))
    phi[domain == 1] = params.phi_anode  # 아노드 전위
    phi[domain == 2] = params.phi_cathode  # 캐소드 전위
    
    # 해수 영역 초기값: 두 전극 사이의 선형 보간
    phi[domain == 0] = (params.phi_anode + params.phi_cathode) / 2
    
    # ----- 부식 상태 배열 초기화 -----
    # 각 아노드 픽셀이 얼마나 부식되었는지 추적 (0: 완전, 1: 완전 부식)
    erosion_state = np.zeros((params.ny, params.nx))
    
    return domain, phi, erosion_state

# ==============================================================================
# 3. 라플라스 방정식 풀이 (유한 차분법 + SOR)
# ==============================================================================

def solve_laplace_fdm(phi, domain, params):
    """
    유한 차분법(FDM)과 SOR 방법을 사용하여 라플라스 방정식을 풉니다.
    
    라플라스 방정식: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² = 0
    
    이산화 형태 (2차 중심 차분):
    φ(i,j) = (1/4)[φ(i+1,j) + φ(i-1,j) + φ(i,j+1) + φ(i,j-1)]
    
    경계 조건:
    - Dirichlet: 아노드, 캐소드 표면에서 고정 전위
    - Neumann: 도메인 경계에서 ∂φ/∂n = 0 (절연 조건)
    
    SOR(Successive Over-Relaxation):
    φ_new = φ_old + ω(φ_gauss_seidel - φ_old)
    여기서 ω = 1.0~2.0 (최적값은 약 1.7~1.9)
    
    Args:
        phi: 현재 전위 분포
        domain: 도메인 배열
        params: 시뮬레이션 파라미터
        
    Returns:
        phi: 수렴된 전위 분포
        converged: 수렴 여부
    """
    phi = phi.copy()
    omega = params.omega
    
    for iteration in range(params.max_iter):
        phi_old = phi.copy()
        max_change = 0.0
        
        # 내부 점들에 대해 반복
        for i in range(1, params.ny - 1):
            for j in range(1, params.nx - 1):
                # 아노드나 캐소드 영역은 고정 전위 (Dirichlet 조건)
                if domain[i, j] != 0:
                    continue
                
                # 5점 스텐실을 사용한 라플라시안 이산화
                # 인접 셀이 전극인 경우와 해수인 경우를 구분
                phi_neighbors = 0.0
                n_neighbors = 0
                
                # 상하좌우 이웃 확인
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in neighbors:
                    if 0 <= ni < params.ny and 0 <= nj < params.nx:
                        phi_neighbors += phi[ni, nj]
                        n_neighbors += 1
                
                if n_neighbors > 0:
                    # Gauss-Seidel 업데이트
                    phi_gs = phi_neighbors / n_neighbors
                    # SOR 적용
                    phi_new = phi[i, j] + omega * (phi_gs - phi[i, j])
                    
                    change = abs(phi_new - phi[i, j])
                    if change > max_change:
                        max_change = change
                    
                    phi[i, j] = phi_new
        
        # ----- Neumann 경계 조건 적용 (절연) -----
        # ∂φ/∂n = 0 → 경계에서 법선 방향 기울기 0
        phi[0, :] = phi[1, :]  # 상단 경계
        phi[-1, :] = phi[-2, :]  # 하단 경계
        phi[:, 0] = phi[:, 1]  # 좌측 경계
        phi[:, -1] = phi[:, -2]  # 우측 경계
        
        # 전극 전위 재설정 (Dirichlet 조건 유지)
        phi[domain == 1] = params.phi_anode
        phi[domain == 2] = params.phi_cathode
        
        # 수렴 판정
        if max_change < params.tolerance:
            return phi, True
    
    return phi, False

# ==============================================================================
# 4. 전류 밀도 및 플럭스 계산
# ==============================================================================

def calculate_current_density(phi, domain, params):
    """
    전위 분포로부터 전류 밀도를 계산합니다.
    
    옴의 법칙: j = -σ∇φ
    
    전류 밀도 성분:
    - jx = -σ ∂φ/∂x
    - jy = -σ ∂φ/∂y
    - |j| = √(jx² + jy²)
    
    Args:
        phi: 전위 분포
        domain: 도메인 배열
        params: 시뮬레이션 파라미터
        
    Returns:
        jx, jy: x, y 방향 전류 밀도 성분 [A/m²]
        j_magnitude: 전류 밀도 크기 [A/m²]
    """
    # 중심 차분을 사용한 전위 기울기 계산
    # ∂φ/∂x ≈ (φ(i,j+1) - φ(i,j-1)) / (2Δx)
    grad_phi_x = np.zeros_like(phi)
    grad_phi_y = np.zeros_like(phi)
    
    # 내부 점에서 중심 차분
    grad_phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * params.dx)
    grad_phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * params.dy)
    
    # 경계에서 전진/후진 차분
    grad_phi_x[:, 0] = (phi[:, 1] - phi[:, 0]) / params.dx
    grad_phi_x[:, -1] = (phi[:, -1] - phi[:, -2]) / params.dx
    grad_phi_y[0, :] = (phi[1, :] - phi[0, :]) / params.dy
    grad_phi_y[-1, :] = (phi[-1, :] - phi[-2, :]) / params.dy
    
    # 전류 밀도 계산 (j = -σ∇φ)
    jx = -params.sigma * grad_phi_x
    jy = -params.sigma * grad_phi_y
    j_magnitude = np.sqrt(jx**2 + jy**2)
    
    return jx, jy, j_magnitude

def calculate_anode_surface_current(domain, j_magnitude, params):
    """
    아노드 표면에서의 전류를 계산합니다.
    
    아노드 표면: 아노드(domain=1)와 해수(domain=0)가 접하는 경계
    
    유효 표면적(Active Surface Area):
    - 해수와 직접 접촉하는 아노드 픽셀만 부식에 참여
    - 아노드 내부 픽셀은 전류가 흐르지 않음
    
    Args:
        domain: 도메인 배열
        j_magnitude: 전류 밀도 크기
        params: 시뮬레이션 파라미터
        
    Returns:
        surface_current: 각 아노드 표면 픽셀의 전류 [A/m²]
        surface_mask: 아노드 표면 마스크
        total_current: 총 전류량 [A]
    """
    # 아노드 표면 검출: 아노드 픽셀 중 해수와 인접한 픽셀
    anode_mask = (domain == 1)
    
    # 해수 영역을 확장하여 표면 검출
    # 구조 요소를 사용한 팽창(dilation)으로 해수와 접하는 영역 찾기
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])  # 4-연결 구조
    
    seawater_mask = (domain == 0)
    seawater_dilated = ndimage.binary_dilation(seawater_mask, structure)
    
    # 아노드 표면 = 아노드 AND (팽창된 해수)
    surface_mask = anode_mask & seawater_dilated
    
    # 표면에서의 전류 밀도
    surface_current = np.zeros_like(j_magnitude)
    
    # 각 표면 픽셀에서 해수 방향으로의 전류 계산
    for i in range(params.ny):
        for j in range(params.nx):
            if surface_mask[i, j]:
                # 인접 해수 픽셀의 전류 밀도 평균
                neighbors_j = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < params.ny and 0 <= nj < params.nx:
                        if domain[ni, nj] == 0:  # 해수
                            neighbors_j.append(j_magnitude[ni, nj])
                
                if neighbors_j:
                    surface_current[i, j] = np.mean(neighbors_j)
    
    # 총 전류량 계산 [A]
    # 각 픽셀의 면적: dx × dy [m²]
    pixel_area = params.dx * params.dy
    total_current = np.sum(surface_current[surface_mask]) * pixel_area
    
    return surface_current, surface_mask, total_current

# ==============================================================================
# 5. 아노드 형상 업데이트 (패러데이 법칙)
# ==============================================================================

def update_anode_geometry(domain, erosion_state, surface_current, surface_mask, params):
    """
    패러데이 법칙을 적용하여 아노드 형상을 업데이트합니다.
    
    패러데이 법칙:
    - 부식 속도: dm/dt = M·I / (z·F)
    - 여기서:
      - m: 부식된 질량 [kg]
      - M: 몰질량 [kg/mol]
      - I: 전류 [A]
      - z: 전자 수
      - F: 패러데이 상수 [C/mol]
    
    부식 깊이 계산:
    - Δh = (M · j · Δt) / (z · F · ρ)
    - 여기서:
      - j: 전류 밀도 [A/m²]
      - Δt: 시간 스텝 [s]
      - ρ: 밀도 [kg/m³]
    
    Args:
        domain: 도메인 배열
        erosion_state: 각 픽셀의 부식 상태 (0~1)
        surface_current: 표면 전류 밀도
        surface_mask: 아노드 표면 마스크
        params: 시뮬레이션 파라미터
        
    Returns:
        domain: 업데이트된 도메인
        erosion_state: 업데이트된 부식 상태
        mass_loss: 이번 스텝에서 손실된 질량 [kg]
    """
    domain = domain.copy()
    erosion_state = erosion_state.copy()
    
    # 부식 깊이 계산 상수
    # Δh / Δt = M · j / (z · F · ρ) [m/s per A/m²]
    erosion_rate_factor = (params.M_anode * params.dt) / \
                          (params.z_anode * params.F * params.rho_anode)
    
    mass_loss = 0.0
    pixel_volume = params.dx * params.dy * params.dx  # 픽셀 부피 [m³] (정육면체 가정)
    
    # 각 표면 픽셀에 대해 부식 진행
    for i in range(params.ny):
        for j in range(params.nx):
            if surface_mask[i, j] and domain[i, j] == 1:
                # 이 픽셀의 전류 밀도에 의한 부식 깊이 [m]
                j_local = surface_current[i, j]
                erosion_depth = erosion_rate_factor * j_local
                
                # 부식 상태 업데이트 (정규화: 픽셀 크기 대비)
                erosion_increment = erosion_depth / params.dx
                erosion_state[i, j] += erosion_increment
                
                # 부식 상태가 임계값을 초과하면 픽셀 제거
                if erosion_state[i, j] >= params.erosion_threshold:
                    # 질량 손실 계산
                    remaining_fraction = 1.0 - (erosion_state[i, j] - erosion_increment)
                    mass_loss += pixel_volume * params.rho_anode * remaining_fraction
                    
                    # 픽셀을 해수로 변환
                    domain[i, j] = 0
                    erosion_state[i, j] = 0.0
    
    return domain, erosion_state, mass_loss

# ==============================================================================
# 6. 시뮬레이션 메인 루프
# ==============================================================================

def run_simulation(params):
    """
    갈바닉 부식 시뮬레이션의 메인 루프를 실행합니다.
    
    시뮬레이션 흐름:
    1. 도메인 초기화
    2. 각 시간 스텝에서:
       a. 라플라스 방정식 풀이 → 전위 분포
       b. 전류 밀도 계산
       c. 표면 전류 및 총 전류량 계산
       d. 패러데이 법칙으로 형상 업데이트
    3. 결과 저장 및 시각화
    
    Args:
        params: 시뮬레이션 파라미터
        
    Returns:
        history: 시뮬레이션 이력 딕셔너리
    """
    print("=" * 60)
    print("해양 환경 희생 양극 갈바닉 부식 시뮬레이션")
    print("=" * 60)
    print(f"그리드 크기: {params.nx} × {params.ny}")
    print(f"해수 전도도: {params.sigma} S/m")
    print(f"아노드 전위: {params.phi_anode} V")
    print(f"캐소드 전위: {params.phi_cathode} V")
    print(f"시간 스텝 수: {params.n_timesteps}")
    print(f"가속 계수: {params.acceleration_factor:.1e}")
    print("=" * 60)
    
    # 초기화
    domain, phi, erosion_state = initialize_domain(params)
    
    # 결과 저장용 리스트
    history = {
        'domains': [domain.copy()],
        'phis': [phi.copy()],
        'total_currents': [],
        'anode_areas': [],
        'mass_losses': [],
        'time_steps': []
    }
    
    # 초기 아노드 면적
    initial_anode_area = np.sum(domain == 1) * params.dx * params.dy
    
    cumulative_time = 0.0
    cumulative_mass_loss = 0.0
    
    for step in range(params.n_timesteps):
        # ----- Step 1: 라플라스 방정식 풀이 -----
        phi, converged = solve_laplace_fdm(phi, domain, params)
        
        if not converged and step % 10 == 0:
            print(f"  경고: 스텝 {step}에서 라플라스 방정식이 수렴하지 않음")
        
        # ----- Step 2: 전류 밀도 계산 -----
        jx, jy, j_magnitude = calculate_current_density(phi, domain, params)
        
        # ----- Step 3: 표면 전류 및 총 전류량 계산 -----
        surface_current, surface_mask, total_current = \
            calculate_anode_surface_current(domain, j_magnitude, params)
        
        # ----- Step 4: 아노드 형상 업데이트 -----
        domain, erosion_state, mass_loss = \
            update_anode_geometry(domain, erosion_state, surface_current, 
                                  surface_mask, params)
        
        # ----- 결과 기록 -----
        cumulative_time += params.dt / params.acceleration_factor  # 실제 시간 [s]
        cumulative_mass_loss += mass_loss
        current_anode_area = np.sum(domain == 1) * params.dx * params.dy
        
        history['domains'].append(domain.copy())
        history['phis'].append(phi.copy())
        history['total_currents'].append(total_current)
        history['anode_areas'].append(current_anode_area)
        history['mass_losses'].append(cumulative_mass_loss)
        history['time_steps'].append(cumulative_time / (3600 * 24))  # 일(day) 단위
        
        # 진행 상황 출력
        if step % 20 == 0 or step == params.n_timesteps - 1:
            area_ratio = current_anode_area / initial_anode_area * 100
            print(f"스텝 {step:3d}/{params.n_timesteps}: "
                  f"총 전류 = {total_current*1000:.3f} mA, "
                  f"아노드 면적 = {area_ratio:.1f}%, "
                  f"시간 = {cumulative_time/(3600*24):.1f}일")
        
        # 아노드가 완전히 소모되면 종료
        if np.sum(domain == 1) == 0:
            print("\n아노드가 완전히 소모되었습니다!")
            break
    
    print("=" * 60)
    print("시뮬레이션 완료!")
    print(f"최종 아노드 면적: {current_anode_area/initial_anode_area*100:.1f}%")
    print(f"총 질량 손실: {cumulative_mass_loss*1000:.2f} g")
    print("=" * 60)
    
    return history

# ==============================================================================
# 7. 시각화 함수
# ==============================================================================

def create_animation(history, params, save_path='galvanic_corrosion_animation.gif'):
    """
    아노드 부식 과정의 애니메이션을 생성합니다.
    
    색상 구분:
    - 파란색: 해수 (Electrolyte)
    - 주황색: 아노드 (Sacrificial Anode)
    - 회색: 캐소드 (Cathode/Hull)
    
    Args:
        history: 시뮬레이션 이력
        params: 시뮬레이션 파라미터
        save_path: 저장 경로
    """
    # 커스텀 컬러맵 생성
    colors = ['#1E88E5', '#FF8C00', '#757575']  # 해수, 아노드, 캐소드
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('해양 환경 희생 양극 갈바닉 부식 시뮬레이션', fontsize=14, fontweight='bold')
    
    # 범례 패치
    legend_patches = [
        mpatches.Patch(color='#1E88E5', label='해수 (Seawater)'),
        mpatches.Patch(color='#FF8C00', label='아노드 (Anode)'),
        mpatches.Patch(color='#757575', label='캐소드 (Cathode)')
    ]
    
    def animate(frame):
        # 매 프레임 클리어
        for ax in axes:
            ax.clear()
        
        # ----- 왼쪽: 도메인 형상 -----
        domain = history['domains'][frame]
        axes[0].imshow(domain, cmap=cmap, vmin=0, vmax=2, origin='lower',
                       extent=[0, params.nx*params.dx*100, 0, params.ny*params.dy*100])
        axes[0].set_xlabel('X [cm]', fontsize=11)
        axes[0].set_ylabel('Y [cm]', fontsize=11)
        axes[0].set_title(f'도메인 형상 (스텝 {frame})', fontsize=12)
        axes[0].legend(handles=legend_patches, loc='upper right', fontsize=9)
        
        # 그리드 라인
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # ----- 오른쪽: 전위 분포 -----
        phi = history['phis'][frame]
        im = axes[1].imshow(phi, cmap='RdYlBu_r', origin='lower',
                           extent=[0, params.nx*params.dx*100, 0, params.ny*params.dy*100],
                           vmin=params.phi_anode, vmax=params.phi_cathode)
        axes[1].set_xlabel('X [cm]', fontsize=11)
        axes[1].set_ylabel('Y [cm]', fontsize=11)
        axes[1].set_title(f'전위 분포 φ [V]', fontsize=12)
        
        # 컬러바
        if frame == 0:
            plt.colorbar(im, ax=axes[1], label='전위 [V]')
        
        # 시간 정보 표시
        if frame > 0 and frame <= len(history['time_steps']):
            time_day = history['time_steps'][frame-1]
            fig.text(0.5, 0.02, f'경과 시간: {time_day:.1f}일 (가속됨)', 
                    ha='center', fontsize=11, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        return axes
    
    # 애니메이션 생성
    n_frames = len(history['domains'])
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)
    
    # 저장
    print(f"애니메이션 저장 중: {save_path}")
    anim.save(save_path, writer='pillow', fps=10, dpi=100)
    print("애니메이션 저장 완료!")
    
    plt.close()
    return anim

def plot_results(history, params, save_path='galvanic_corrosion_results.png'):
    """
    시뮬레이션 결과 그래프를 생성합니다.
    
    그래프:
    1. 시간에 따른 총 전류량 변화
    2. 시간에 따른 아노드 면적 변화
    3. 시간에 따른 누적 질량 손실
    4. 최종 상태 비교 (초기 vs 최종)
    
    Args:
        history: 시뮬레이션 이력
        params: 시뮬레이션 파라미터
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('갈바닉 부식 시뮬레이션 결과 분석', fontsize=14, fontweight='bold')
    
    time_days = history['time_steps']
    
    # ----- 그래프 1: 시간에 따른 총 전류량 -----
    ax1 = axes[0, 0]
    currents_mA = [c * 1000 for c in history['total_currents']]  # mA 단위
    ax1.plot(time_days, currents_mA, 'b-', linewidth=2, marker='o', 
             markersize=3, label='총 전류량')
    ax1.set_xlabel('시간 [일]', fontsize=11)
    ax1.set_ylabel('총 전류량 [mA]', fontsize=11)
    ax1.set_title('시간에 따른 총 전류량 변화', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 추세선 추가
    if len(time_days) > 2:
        z = np.polyfit(time_days, currents_mA, 1)
        p = np.poly1d(z)
        ax1.plot(time_days, p(time_days), 'r--', alpha=0.7, label='추세선')
    
    # ----- 그래프 2: 시간에 따른 아노드 면적 -----
    ax2 = axes[0, 1]
    initial_area = history['anode_areas'][0] if history['anode_areas'] else 1
    area_ratio = [a / initial_area * 100 for a in history['anode_areas']]
    ax2.plot(time_days, area_ratio, 'g-', linewidth=2, marker='s', 
             markersize=3, label='아노드 면적')
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% 수명')
    ax2.set_xlabel('시간 [일]', fontsize=11)
    ax2.set_ylabel('아노드 면적 [%]', fontsize=11)
    ax2.set_title('시간에 따른 아노드 면적 변화', fontsize=12)
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # ----- 그래프 3: 시간에 따른 누적 질량 손실 -----
    ax3 = axes[1, 0]
    mass_g = [m * 1000 for m in history['mass_losses']]  # g 단위
    ax3.plot(time_days, mass_g, 'm-', linewidth=2, marker='^', 
             markersize=3, label='누적 질량 손실')
    ax3.set_xlabel('시간 [일]', fontsize=11)
    ax3.set_ylabel('질량 손실 [g]', fontsize=11)
    ax3.set_title('시간에 따른 누적 질량 손실', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # ----- 그래프 4: 초기 vs 최종 형상 비교 -----
    ax4 = axes[1, 1]
    colors = ['#1E88E5', '#FF8C00', '#757575']
    cmap = ListedColormap(colors)
    
    # 최종 상태만 표시
    final_domain = history['domains'][-1]
    ax4.imshow(final_domain, cmap=cmap, vmin=0, vmax=2, origin='lower',
               extent=[0, params.nx*params.dx*100, 0, params.ny*params.dy*100])
    ax4.set_xlabel('X [cm]', fontsize=11)
    ax4.set_ylabel('Y [cm]', fontsize=11)
    ax4.set_title('최종 도메인 형상', fontsize=12)
    
    # 초기 아노드 윤곽선
    initial_domain = history['domains'][0]
    initial_anode = (initial_domain == 1).astype(float)
    ax4.contour(initial_anode, levels=[0.5], colors='red', linewidths=2,
                extent=[0, params.nx*params.dx*100, 0, params.ny*params.dy*100],
                linestyles='--')
    
    # 범례
    legend_patches = [
        mpatches.Patch(color='#1E88E5', label='해수'),
        mpatches.Patch(color='#FF8C00', label='아노드 (현재)'),
        mpatches.Patch(color='#757575', label='캐소드'),
        mpatches.Patch(facecolor='none', edgecolor='red', 
                       linestyle='--', label='아노드 (초기)')
    ]
    ax4.legend(handles=legend_patches, loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"결과 그래프 저장: {save_path}")
    plt.close()

def plot_current_vs_time(history, save_path='current_vs_time.png'):
    """
    시간에 따른 총 전류량 변화 그래프 (요구사항 명시)
    
    아노드가 작아짐에 따라 전류량이 줄어드는 경향을 명확히 보여줍니다.
    
    Args:
        history: 시뮬레이션 이력
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_days = history['time_steps']
    currents_mA = [c * 1000 for c in history['total_currents']]
    
    # 메인 플롯
    ax.plot(time_days, currents_mA, 'b-', linewidth=2.5, marker='o', 
            markersize=4, markerfacecolor='white', markeredgecolor='blue',
            markeredgewidth=1.5, label='총 전류량')
    
    # 영역 채우기
    ax.fill_between(time_days, currents_mA, alpha=0.3, color='blue')
    
    # 스타일링
    ax.set_xlabel('시간 (Time Step) [일]', fontsize=12, fontweight='bold')
    ax.set_ylabel('총 전류량 (Total Current) [mA]', fontsize=12, fontweight='bold')
    ax.set_title('시간에 따른 총 전류량 변화\n(아노드 유효 표면적 감소에 따른 전류 감소)', 
                fontsize=14, fontweight='bold')
    
    # 그리드
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 통계 정보 박스
    initial_current = currents_mA[0] if currents_mA else 0
    final_current = currents_mA[-1] if currents_mA else 0
    reduction_pct = (1 - final_current / initial_current) * 100 if initial_current > 0 else 0
    
    textstr = '\n'.join([
        f'초기 전류: {initial_current:.2f} mA',
        f'최종 전류: {final_current:.2f} mA',
        f'전류 감소율: {reduction_pct:.1f}%'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # 추세선
    if len(time_days) > 2:
        z = np.polyfit(time_days, currents_mA, 2)  # 2차 다항식
        p = np.poly1d(z)
        ax.plot(time_days, p(time_days), 'r--', linewidth=1.5, 
                alpha=0.7, label='추세선 (2차)')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim([0, max(time_days) * 1.02])
    ax.set_ylim([0, max(currents_mA) * 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"전류-시간 그래프 저장: {save_path}")
    plt.close()

# ==============================================================================
# 8. 메인 실행
# ==============================================================================

if __name__ == "__main__":
    # 시뮬레이션 파라미터 설정
    params = SimulationParameters()
    
    # 시뮬레이션 실행
    history = run_simulation(params)
    
    # 결과 시각화
    print("\n결과 시각화 생성 중...")
    
    # 1. 애니메이션 생성
    create_animation(history, params, 'galvanic_corrosion_animation.gif')
    
    # 2. 종합 결과 그래프
    plot_results(history, params, 'galvanic_corrosion_results.png')
    
    # 3. 전류-시간 그래프 (요구사항 명시)
    plot_current_vs_time(history, 'current_vs_time.png')
    
    print("\n모든 시각화 완료!")
    print("생성된 파일:")
    print("  - galvanic_corrosion_animation.gif (애니메이션)")
    print("  - galvanic_corrosion_results.png (종합 결과)")
    print("  - current_vs_time.png (전류-시간 그래프)")

