#!/usr/bin/env python3
"""
실제 ARC 문제 2a5f8217에 대한 정확한 GLM 프롬프트 예시
"""

def show_real_glm_prompt_example():
    """실제 ARC 문제 2a5f8217에 대한 GLM 프롬프트"""
    
    print("ARC 문제 2a5f8217 분석:")
    print("="*60)
    print("패턴: 입력에서 1(파란색)로 된 connected component가 다른 색상으로 변환됨")
    print("규칙: 각 1로 된 패턴이 그리드의 다른 부분에 있는 특정 색상으로 대체됨")
    print()
    
    # 실제 문제에 맞는 BARC 코드 예시
    barc_code = """
def solve(grid):
    # 문제 분석: 1(파란색) 패턴들이 다른 색상으로 변환됨
    # 각 1로 이루어진 connected component를 찾고, 
    # 다른 색상 패턴과 매칭하여 변환
    
    import numpy as np
    from collections import deque
    
    result = grid.copy()
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    
    # 색상별 패턴 찾기 (1이 아닌 색상들)
    color_patterns = {}
    for color in range(2, 10):
        if np.any(grid == color):
            color_patterns[color] = find_pattern(grid, color)
    
    # 1로 된 connected components 찾기
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 1 and not visited[i][j]:
                component = get_connected_component(grid, i, j, visited)
                # 이 component를 어떤 색으로 바꿀지 결정
                target_color = determine_target_color(component, color_patterns)
                # component를 target_color로 변환
                for ci, cj in component:
                    result[ci][cj] = target_color
    
    return result

def get_connected_component(grid, start_i, start_j, visited):
    # BFS로 connected component 찾기
    component = []
    queue = deque([(start_i, start_j)])
    visited[start_i][start_j] = True
    
    while queue:
        i, j = queue.popleft()
        component.append((i, j))
        
        for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and
                not visited[ni][nj] and grid[ni][nj] == 1):
                visited[ni][nj] = True
                queue.append((ni, nj))
    
    return component
"""
    
    barc_description = "각 파란색(1) connected component를 그리드의 다른 부분에 존재하는 특정 색상으로 변환하는 패턴. 각 component는 위치나 모양에 따라 서로 다른 색상으로 매핑됨."
    
    barc_concepts = "connected_components, color_mapping, pattern_matching, spatial_analysis"
    
    print("\nGLM이 받는 실제 프롬프트:")
    print("="*80)
    
    glm_prompt = f"""Look at these two images:
1. The first image shows the original ARC training examples
2. The second image shows our generated solution applied to the test input

Code Description: "{barc_description}"
Code Concepts: "{barc_concepts}"

<think>
Analyze by comparing the two images:

1. UNDERSTANDING CHECK: Does the description match the pattern shown in the training examples?
   - Look at the training examples in the first image
   - Check if the description accurately explains the transformation pattern
   - Verify if the concepts align with what you observe

2. CALCULATION CHECK: Does the generated output follow the same transformation rule as the training examples?
   - Compare the transformation pattern from training examples
   - Check if the generated output in the second image follows the same logic
   - Look for any calculation or logic errors

3. ANSWER COMPLETENESS: Is the generated output complete and properly formatted?
   - Check if the output provides a definitive solution (not partial)
   - Verify if all necessary transformations are applied
   - Ensure the output format is correct

4. ANSWER CORRECT: Does the transformation logic appear correct based on the pattern?
   - Assess overall correctness of the approach
   - Check consistency with training examples
</think>

<answer>
Provide your evaluation in the following format:

UNDERSTANDING_CHECK: [TRUE/FALSE]
CALCULATION_CHECK: [TRUE/FALSE] 
ANSWER_COMPLETENESS: [TRUE/FALSE]
ANSWER_CORRECT: [TRUE/FALSE]

FEEDBACK: [Brief explanation of your assessment]
</answer>"""
    
    print(glm_prompt)
    
    print("\n" + "="*80)
    print("GLM 응답 예시 (올바른 경우):")
    print("="*80)
    
    correct_response = """<think>
이 두 이미지를 분석해보니:

1. 훈련 예시들을 보면 파란색(1)으로 된 connected component들이 다른 색상으로 변환되는 패턴이 명확히 보입니다. 설명에서 언급한 "connected component 변환"이 정확히 관찰되는 패턴과 일치합니다.

2. 생성된 솔루션에서 테스트 입력의 파란색 component들이 적절한 색상으로 변환되었는지 확인해보니, 훈련 예시의 논리와 일치하는 변환이 적용되었습니다.

3. 출력이 완전한 그리드 형태로 제공되었고, 모든 필요한 변환이 적용되었습니다.

4. 전체적으로 훈련 예시들과 일관된 변환 논리가 적용되었습니다.
</think>

<answer>
UNDERSTANDING_CHECK: TRUE
CALCULATION_CHECK: TRUE
ANSWER_COMPLETENESS: TRUE
ANSWER_CORRECT: TRUE

FEEDBACK: The solution correctly identifies and transforms blue connected components to appropriate colors. The description accurately captures the core pattern of component-based color transformation, and the implementation successfully applies this logic to the test case following the established pattern from training examples.
</answer>"""
    
    print(correct_response)
    
    print("\n" + "="*80)
    print("GLM 응답 예시 (잘못된 경우):")
    print("="*80)
    
    incorrect_response = """<think>
이 두 이미지를 분석해보니:

1. 훈련 예시들을 보면 파란색 셀들이 변환되는 패턴이 있지만, 설명에서 언급한 "connected component" 개념이 실제로는 단순한 색상 대체에 가깝습니다. 설명이 패턴을 과도하게 복잡하게 해석한 것 같습니다.

2. 생성된 솔루션을 보니 일부 파란색 셀들이 잘못된 색상으로 변환되었습니다. 특히 테스트 케이스의 왼쪽 상단 component가 예상과 다른 색상으로 변환되었습니다.

3. 출력 자체는 완전하지만, 변환 논리에 오류가 있습니다.

4. 전체적으로 훈련 예시들과 일관성이 부족합니다.
</think>

<answer>
UNDERSTANDING_CHECK: FALSE
CALCULATION_CHECK: FALSE
ANSWER_COMPLETENESS: TRUE
ANSWER_CORRECT: FALSE

FEEDBACK: While the solution attempts to transform blue components, there are errors in the color mapping logic. Some components are transformed to incorrect colors, and the description over-complicates the actual pattern. The implementation doesn't consistently follow the transformation rules observed in training examples.
</answer>"""
    
    print(incorrect_response)
    
    print("\n" + "="*80)
    print("응답 파싱 및 점수 계산:")
    print("="*80)
    
    print("""
올바른 응답의 경우:
- understanding_check: VerificationResult(passed=True, confidence=0.8)
- calculation_check: VerificationResult(passed=True, confidence=0.8)
- answer_completeness: VerificationResult(passed=True, confidence=0.8)
- answer_correct: VerificationResult(passed=True, confidence=0.8)

점수 계산:
- understanding_check: 0.0 (통과, 패널티 없음)
- calculation_check: 0.0 (통과, 패널티 없음)
- answer_completeness: 0.0 (통과, 패널티 없음)
- answer_correct: 0.0 (통과, 패널티 없음)
총 보상: 0.0 (완벽한 점수)

잘못된 응답의 경우:
- understanding_check: VerificationResult(passed=False, confidence=0.8)
- calculation_check: VerificationResult(passed=False, confidence=0.8)
- answer_completeness: VerificationResult(passed=True, confidence=0.8)
- answer_correct: VerificationResult(passed=False, confidence=0.8)

점수 계산 (가중치: understanding=1.0, calculation=2.0, completeness=2.0, correct=1.0):
- understanding_check: -1.0 * 0.8 = -0.8
- calculation_check: -2.0 * 0.8 = -1.6
- answer_completeness: 0.0 (통과)
- answer_correct: -1.0 * 0.8 = -0.8
총 패널티: -3.2
총 보상: -3.2 / 6.0 = -0.533 (정규화됨)
""")

if __name__ == "__main__":
    show_real_glm_prompt_example()