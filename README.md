# AD Scenario Classifier

실제 자율주행 로그 데이터를 위한 자동화된 시나리오 분류 파이프라인

## 📋 개요

이 프로젝트는 실제 자율주행 차량에서 수집된 로그 데이터를 분석하여 다양한 주행 시나리오를 자동으로 분류하는 파이프라인을 제공합니다. 머신러닝 및 딥러닝 기법을 활용하여 주행 데이터의 패턴을 학습하고 시나리오를 분류합니다.

## ✨ 주요 기능

- **자동 시나리오 분류**: 자율주행 로그 데이터에서 다양한 주행 시나리오를 자동으로 식별 및 분류
- **실시간 처리**: 대용량 로그 데이터를 효율적으로 처리
- **확장 가능한 아키텍처**: 새로운 시나리오 유형 추가 및 모델 개선 용이
- **시각화**: 분류 결과 및 데이터 분석 결과 시각화

## 🔧 요구사항

- Python 3.8 이상
- (추가 요구사항은 프로젝트 진행에 따라 업데이트 예정)

## 📦 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/ad_scenario_classifier.git
cd ad_scenario_classifier

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 🚀 사용 방법

```bash
# 기본 실행 예시
python main.py --input <로그_파일_경로> --output <결과_저장_경로>

# 추가 옵션
python main.py --input data/logs/ --output results/ --model <모델_경로>
```

## 📁 프로젝트 구조

```
ad_scenario_classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── main.py
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── data/
├── results/
└── tests/
```

(프로젝트 구조는 실제 구현에 따라 업데이트 예정)

## 📊 데이터 형식

입력 데이터 형식 및 출력 형식에 대한 설명은 프로젝트 진행에 따라 추가 예정입니다.

## 🤝 기여 방법

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성합니다

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📧 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.

---

**참고**: 이 프로젝트는 현재 개발 중입니다. 문서 및 기능은 지속적으로 업데이트될 예정입니다.
