# IdeaHub Analytics

Firebase Analytics 데이터를 수집하고 분석하는 대시보드입니다.

## 설치 방법

1. Python 3.8 이상 설치
2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. Firebase 서비스 계정 키 설정:
   - Firebase Console에서 서비스 계정 키 다운로드
   - `config/serviceAccountKey.json`에 저장

## 실행 방법

1. 데이터 수집:
```bash
python src/data_collector.py
```

2. 대시보드 실행:
```bash
streamlit run src/dashboard.py
```

## 주요 기능

1. 주요 지표
   - 총 사용자 수
   - 총 이벤트 수
   - 이탈률
   - 포트폴리오 수

2. 이벤트 분석
   - 일별 이벤트 발생 추이
   - 전환 퍼널 분석
   - 시간대별 활동량

3. 포트폴리오 분석
   - 평균 섹션 수
   - 평균 프로젝트 수
   - 공유율

4. 사용자 행동 분석
   - 상위 10개 사용자 여정
   - 사용자 행동 패턴

## 데이터 구조

1. 이벤트 데이터 (`events.csv`)
   - user_id: 사용자 ID
   - event_name: 이벤트 이름
   - timestamp: 발생 시간
   - properties: 이벤트 속성

2. 포트폴리오 데이터 (`portfolios.csv`)
   - id: 포트폴리오 ID
   - user_id: 생성자 ID
   - sections: 섹션 목록
   - projects: 프로젝트 목록
   - created_at: 생성 시간
   - updated_at: 수정 시간
   - shared: 공유 여부

3. 프로젝트 데이터 (`projects.csv`)
   - id: 프로젝트 ID
   - user_id: 생성자 ID
   - title: 제목
   - description: 설명
   - created_at: 생성 시간
   - updated_at: 수정 시간 