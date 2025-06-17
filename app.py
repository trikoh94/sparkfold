import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import streamlit.components.v1 as components

from data_collector import DataCollector
from data_analyzer import DataAnalyzer

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """메트릭 카드 생성"""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color
    )

def check_data_quality(df, name):
    """데이터 품질 체크"""
    if df.empty:
        return {
            'status': '⚠️',
            'message': f'No {name} data available'
        }
    
    total_rows = len(df)
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / total_rows * 100).round(2)
    
    return {
        'status': '✅' if null_percentages.max() < 10 else '⚠️',
        'message': f'{name} data available ({total_rows:,} rows)',
        'null_percentages': null_percentages[null_percentages > 0]
    }

def create_sequences(data, seq_length):
    """시계열 데이터를 시퀀스로 변환"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length, n_features):
    """LSTM 모델 생성"""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_time_series(events_df, target_column='event_count', days_to_predict=30):
    """시계열 예측 수행"""
    # 일별 데이터 집계
    daily_data = events_df.groupby(pd.to_datetime(events_df['timestamp']).dt.date).size().reset_index(name=target_column)
    daily_data.set_index('timestamp', inplace=True)
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_data[[target_column]])
    
    # 시퀀스 생성
    seq_length = 7  # 7일 데이터로 다음 날 예측
    X, y = create_sequences(scaled_data, seq_length)
    
    # 데이터 분할
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 모델 학습
    model = build_lstm_model(seq_length, 1)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # 미래 예측
    last_sequence = scaled_data[-seq_length:]
    future_predictions = []
    
    for _ in range(days_to_predict):
        next_pred = model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    
    # 예측값 역정규화
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # 예측 날짜 생성
    last_date = daily_data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    
    return daily_data, future_dates, future_predictions

def generate_sample_data():
    """샘플 데이터 생성"""
    np.random.seed(42)
    sample_size = 5000
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
    
    # 기본 이벤트 데이터
    events_df = pd.DataFrame({
        'timestamp': np.random.choice(dates, sample_size),
        'user_id': np.random.randint(1, 200, sample_size),
        'event_type': np.random.choice([
            'idea_created', 'project_created', 'portfolio_created', 
            'idea_viewed', 'idea_shared', 'project_updated',
            'portfolio_shared', 'comment_added'
        ], sample_size),
        'session_duration': np.random.randint(60, 7200, sample_size),
        'idea_complexity': np.random.randint(1, 10, sample_size),
        'user_experience': np.random.randint(1, 100, sample_size),
        'interaction_count': np.random.randint(1, 50, sample_size),
    })
    
    # 시계열 패턴 추가
    base_activity = 100  # 기본 활동량
    weekly_pattern = np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 50  # 주간 패턴
    trend = np.linspace(0, 50, len(dates))  # 상승 트렌드
    noise = np.random.normal(0, 10, len(dates))  # 노이즈
    
    daily_activity = base_activity + weekly_pattern + trend + noise
    daily_activity = np.maximum(daily_activity, 0)  # 음수 방지
    
    # 이벤트 수 조정
    for i, date in enumerate(dates):
        date_events = events_df[events_df['timestamp'] == date]
        if len(date_events) > 0:
            target_count = int(daily_activity[i])
            if len(date_events) > target_count:
                events_df = events_df[~((events_df['timestamp'] == date) & 
                                     (events_df.index.isin(date_events.index[target_count:])))]
    
    # 프로젝트 데이터
    projects_df = pd.DataFrame({
        'project_id': range(1, 501),
        'user_id': np.random.randint(1, 200, 500),
        'created_at': np.random.choice(dates, 500),
        'completion_status': np.random.choice(['completed', 'in_progress', 'abandoned'], 500),
        'project_duration': np.random.randint(1, 90, 500),
        'team_size': np.random.randint(1, 8, 500),
        'complexity_score': np.random.randint(1, 10, 500),
        'milestone_count': np.random.randint(1, 10, 500),
    })
    
    # 포트폴리오 데이터
    portfolios_df = pd.DataFrame({
        'portfolio_id': range(1, 301),
        'user_id': np.random.randint(1, 200, 300),
        'created_at': np.random.choice(dates, 300),
        'type': np.random.choice(['personal', 'professional', 'academic', 'startup', 'research'], 300),
        'is_shared': np.random.choice([True, False], 300),
        'view_count': np.random.randint(0, 5000, 300),
        'like_count': np.random.randint(0, 1000, 300),
        'comment_count': np.random.randint(0, 500, 300),
        'last_updated': np.random.choice(dates, 300),
    })
    
    return events_df, projects_df, portfolios_df

def perform_user_clustering(events_df):
    """사용자 행동 클러스터링"""
    # 사용자별 특성 추출
    user_features = events_df.groupby('user_id').agg({
        'session_duration': 'mean',
        'idea_complexity': 'mean',
        'user_experience': 'mean'
    }).reset_index()
    
    # 특성 스케일링
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(user_features.drop('user_id', axis=1))
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_features['cluster'] = kmeans.fit_predict(features_scaled)
    
    return user_features

def predict_conversion(events_df, projects_df):
    """아이디어-프로젝트 전환 예측"""
    # 특성 엔지니어링
    user_events = events_df.groupby('user_id').agg({
        'session_duration': ['mean', 'sum'],
        'idea_complexity': 'mean',
        'user_experience': 'mean'
    }).reset_index()
    
    user_events.columns = ['user_id', 'avg_session_duration', 'total_session_duration', 
                          'avg_idea_complexity', 'user_experience']
    
    # 전환 여부 레이블 생성
    converted_users = projects_df['user_id'].unique()
    user_events['converted'] = user_events['user_id'].isin(converted_users).astype(int)
    
    # 모델 학습
    X = user_events.drop(['user_id', 'converted'], axis=1)
    y = user_events['converted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def predict_engagement(events_df):
    """사용자 참여도 예측"""
    # 사용자별 참여도 계산
    user_engagement = events_df.groupby('user_id').agg({
        'session_duration': 'sum',
        'idea_complexity': 'mean',
        'user_experience': 'mean'
    }).reset_index()
    
    # 참여도 점수 계산 (예: 세션 시간의 가중 평균)
    user_engagement['engagement_score'] = (
        user_engagement['session_duration'] * 0.5 +
        user_engagement['idea_complexity'] * 0.3 +
        user_engagement['user_experience'] * 0.2
    )
    
    # 모델 학습
    X = user_engagement.drop(['user_id', 'engagement_score'], axis=1)
    y = user_engagement['engagement_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def main():
    st.set_page_config(page_title="IdeaHub Analytics", layout="wide")
    st.title('IdeaHub Analytics Dashboard')
    
    # 데이터 로드
    try:
        collector = DataCollector()
        events_df = collector.collect_events()
        portfolios_df = collector.collect_portfolio_data()
        projects_df = collector.collect_project_data()
    except Exception as e:
        st.warning("Firebase connection failed. Using sample data instead.")
        events_df, projects_df, portfolios_df = generate_sample_data()
    
    # 데이터 품질 체크 섹션
    st.sidebar.header("Data Quality Check")
    events_quality = check_data_quality(events_df, "Events")
    portfolios_quality = check_data_quality(portfolios_df, "Portfolios")
    projects_quality = check_data_quality(projects_df, "Projects")
    
    st.sidebar.subheader("Data Status")
    st.sidebar.write(f"{events_quality['status']} {events_quality['message']}")
    st.sidebar.write(f"{portfolios_quality['status']} {portfolios_quality['message']}")
    st.sidebar.write(f"{projects_quality['status']} {projects_quality['message']}")
    
    # 데이터 수집 기간 표시
    if not events_df.empty and 'timestamp' in events_df.columns:
        start_date = pd.to_datetime(events_df['timestamp']).min().date()
        end_date = pd.to_datetime(events_df['timestamp']).max().date()
        st.sidebar.subheader("Data Collection Period")
        st.sidebar.write(f"From: {start_date}")
        st.sidebar.write(f"To: {end_date}")
        st.sidebar.write(f"Total Days: {(end_date - start_date).days + 1}")
    
    # 샘플 데이터 생성 옵션
    if st.sidebar.checkbox("Regenerate Sample Data"):
        st.sidebar.warning("This will regenerate sample data for demonstration purposes")
        if st.sidebar.button("Generate"):
            events_df, projects_df, portfolios_df = generate_sample_data()
            st.sidebar.success("Sample data regenerated!")
    
    # 데이터가 비어있는 경우 처리
    if events_df.empty:
        st.warning("No event data available. Please check Firebase connection and data.")
        return
    
    # 분석기 초기화
    analyzer = DataAnalyzer(events_df)
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Mathematical Analysis", "User Behavior", 
        "Growth Metrics", "Machine Learning", "Architecture"
    ])
    
    with tab1:
        st.header('Key Performance Indicators')
        
        # KPI 메트릭스
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_users = len(events_df['user_id'].unique()) if 'user_id' in events_df.columns else 0
            create_metric_card("Total Users", f"{total_users:,}")
        
        with col2:
            total_ideas = len(events_df[events_df['event_type'] == 'idea_created']) if 'event_type' in events_df.columns else 0
            create_metric_card("Total Ideas", f"{total_ideas:,}")
        
        with col3:
            total_projects = len(projects_df) if not projects_df.empty else 0
            create_metric_card("Total Projects", f"{total_projects:,}")
        
        with col4:
            total_portfolios = len(portfolios_df) if not portfolios_df.empty else 0
            create_metric_card("Total Portfolios", f"{total_portfolios:,}")
        
        # 전환율 차트
        st.subheader('Conversion Funnel')
        funnel_data = {
            'Stage': ['Ideas', 'Projects', 'Portfolios', 'Shared'],
            'Count': [
                len(events_df[events_df['event_type'] == 'idea_created']) if 'event_type' in events_df.columns else 0,
                len(projects_df) if not projects_df.empty else 0,
                len(portfolios_df) if not portfolios_df.empty else 0,
                len(portfolios_df[portfolios_df['is_shared'] == True]) if not portfolios_df.empty and 'is_shared' in portfolios_df.columns else 0
            ]
        }
        funnel_df = pd.DataFrame(funnel_data)
        
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_df['Stage'],
            x=funnel_df['Count'],
            textinfo="value+percent initial"
        ))
        fig_funnel.update_layout(title='Conversion Funnel Analysis')
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        # 시간대별 활동 분석
        st.subheader('Activity by Time')
        if 'timestamp' in events_df.columns:
            events_df['hour'] = pd.to_datetime(events_df['timestamp']).dt.hour
            hourly_activity = events_df.groupby('hour').size().reset_index(name='count')
            
            fig_hourly = px.bar(
                hourly_activity,
                x='hour',
                y='count',
                title='Hourly Activity Distribution',
                labels={'hour': 'Hour of Day', 'count': 'Number of Events'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.warning("Timestamp data not available for hourly analysis")
        
        # 포트폴리오 타입 분포
        st.subheader('Portfolio Type Distribution')
        if not portfolios_df.empty and 'type' in portfolios_df.columns:
            portfolio_types = portfolios_df['type'].value_counts()
            fig_portfolio = px.pie(
                values=portfolio_types.values,
                names=portfolio_types.index,
                title='Portfolio Types'
            )
            st.plotly_chart(fig_portfolio, use_container_width=True)
        else:
            st.warning("Portfolio type data not available")
    
    with tab2:
        st.header('Mathematical Analysis')
        
        if not events_df.empty:
            # 1. 확률론적 분석
            st.subheader("1. Probability Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # 이탈률 (베이지안 업데이트)
                bounce_rate = analyzer.get_bounce_rate()
                create_metric_card("Bounce Rate", f"{bounce_rate:.2%}")
                
                # 전환율 (조건부 확률)
                conversion_rate = analyzer.get_conversion_rate()
                create_metric_card("Conversion Rate", f"{conversion_rate:.2%}")
            
            with col2:
                # 프로젝트 완료율 (베르누이 분포)
                completion_rate = analyzer.get_completion_rate()
                create_metric_card("Project Completion Rate", f"{completion_rate:.2%}")
                
                # 아이디어-프로젝트 전환율
                idea_to_project_rate = analyzer.get_idea_to_project_rate()
                create_metric_card("Idea to Project Rate", f"{idea_to_project_rate:.2%}")
            
            # 2. 시계열 분석
            st.subheader("2. Time Series Analysis")
            
            # 일별 활동 추이
            events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date
            daily_activity = events_df.groupby('date').size().reset_index(name='count')
            
            fig_daily = px.line(
                daily_activity,
                x='date',
                y='count',
                title='Daily Activity Trend',
                labels={'date': 'Date', 'count': 'Number of Events'}
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # 3. 상관관계 분석
            st.subheader("3. Correlation Analysis")
            
            if not portfolios_df.empty:
                # 조회수와 좋아요 수의 상관관계
                fig_correlation = px.scatter(
                    portfolios_df,
                    x='view_count',
                    y='like_count',
                    color='type',
                    title='Correlation: Views vs Likes by Portfolio Type',
                    labels={'view_count': 'Number of Views', 'like_count': 'Number of Likes'}
                )
                st.plotly_chart(fig_correlation, use_container_width=True)
                
                # 댓글 수와 좋아요 수의 상관관계
                fig_correlation2 = px.scatter(
                    portfolios_df,
                    x='comment_count',
                    y='like_count',
                    color='type',
                    title='Correlation: Comments vs Likes by Portfolio Type',
                    labels={'comment_count': 'Number of Comments', 'like_count': 'Number of Likes'}
                )
                st.plotly_chart(fig_correlation2, use_container_width=True)
    
    with tab3:
        st.header('User Behavior Analysis')
        
        if not events_df.empty:
            # 1. 사용자 행동 흐름
            st.subheader("1. User Journey Analysis")
            
            # 이벤트 시퀀스 분석
            event_sequences = analyzer.get_event_sequences()
            if not event_sequences.empty:
                fig_sequences = px.sunburst(
                    event_sequences,
                    path=['sequence'],
                    values='count',
                    title='Top User Event Sequences'
                )
                st.plotly_chart(fig_sequences, use_container_width=True)
            
            # 2. 사용자 세그먼트
            st.subheader("2. User Segments")
            
            # 활동 수준별 사용자 분류
            user_activity = events_df.groupby('user_id').size().reset_index(name='activity_count')
            user_activity['segment'] = pd.qcut(
                user_activity['activity_count'],
                q=4,
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            fig_segments = px.pie(
                user_activity,
                names='segment',
                title='User Activity Segments'
            )
            st.plotly_chart(fig_segments, use_container_width=True)
            
            # 3. 사용자 행동 패턴
            st.subheader("3. User Behavior Patterns")
            
            # 이벤트 타입별 분포
            event_types = events_df['event_type'].value_counts()
            fig_events = px.bar(
                x=event_types.index,
                y=event_types.values,
                title='Event Type Distribution',
                labels={'x': 'Event Type', 'y': 'Count'}
            )
            st.plotly_chart(fig_events, use_container_width=True)
    
    with tab4:
        st.header('Growth Metrics')
        
        if not events_df.empty:
            # 1. 성장 지표
            st.subheader("1. Growth Metrics")
            
            # 일별 신규 사용자
            events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date
            new_users = events_df.groupby('date')['user_id'].nunique().reset_index(name='new_users')
            
            fig_new_users = px.line(
                new_users,
                x='date',
                y='new_users',
                title='Daily New Users',
                labels={'date': 'Date', 'new_users': 'Number of New Users'}
            )
            st.plotly_chart(fig_new_users, use_container_width=True)
            
            # 2. 참여도 지표
            st.subheader("2. Engagement Metrics")
            
            # 사용자당 평균 이벤트 수
            user_engagement = events_df.groupby('user_id').size().reset_index(name='event_count')
            avg_events = user_engagement['event_count'].mean()
            
            create_metric_card(
                "Average Events per User",
                f"{avg_events:.1f}",
                delta=f"{avg_events - user_engagement['event_count'].median():.1f}",
                delta_color="normal"
            )
            
            # 3. 전환율 추이
            st.subheader("3. Conversion Rate Trends")
            
            # 일별 전환율 계산
            daily_conversion = events_df.groupby('date').apply(
                lambda x: len(x[x['event_type'] == 'project_created']) / len(x[x['event_type'] == 'idea_created'])
                if len(x[x['event_type'] == 'idea_created']) > 0 else 0
            ).reset_index(name='conversion_rate')
            
            fig_conversion = px.line(
                daily_conversion,
                x='date',
                y='conversion_rate',
                title='Daily Conversion Rate Trend',
                labels={'date': 'Date', 'conversion_rate': 'Conversion Rate'}
            )
            st.plotly_chart(fig_conversion, use_container_width=True)
    
    with tab5:
        st.header('Machine Learning Analysis')
        
        if not events_df.empty:
            # 1. 사용자 클러스터링
            st.subheader("1. User Behavior Clustering")
            user_clusters = perform_user_clustering(events_df)
            
            fig_clusters = px.scatter_3d(
                user_clusters,
                x='session_duration',
                y='idea_complexity',
                z='user_experience',
                color='cluster',
                title='User Behavior Clusters'
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # 2. 전환 예측
            st.subheader("2. Conversion Prediction")
            conversion_model, X_test, y_test = predict_conversion(events_df, projects_df)
            
            # 모델 성능 표시
            accuracy = conversion_model.score(X_test, y_test)
            create_metric_card("Conversion Prediction Accuracy", f"{accuracy:.2%}")
            
            # 특성 중요도 시각화
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': conversion_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='feature',
                y='importance',
                title='Feature Importance for Conversion Prediction'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # 3. 참여도 예측
            st.subheader("3. Engagement Prediction")
            engagement_model, X_test, y_test = predict_engagement(events_df)
            
            # 모델 성능 표시
            r2_score = engagement_model.score(X_test, y_test)
            create_metric_card("Engagement Prediction R² Score", f"{r2_score:.2f}")
            
            # 예측값 vs 실제값 시각화
            y_pred = engagement_model.predict(X_test)
            fig_engagement = px.scatter(
                x=y_test,
                y=y_pred,
                title='Predicted vs Actual Engagement Scores',
                labels={'x': 'Actual', 'y': 'Predicted'}
            )
            fig_engagement.add_scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Perfect Prediction'
            )
            st.plotly_chart(fig_engagement, use_container_width=True)
            
            # 4. 시계열 예측
            st.subheader("4. Time Series Prediction")
            
            # 예측 수행
            daily_data, future_dates, future_predictions = predict_time_series(events_df)
            
            # 시각화
            fig_forecast = go.Figure()
            
            # 실제 데이터
            fig_forecast.add_trace(go.Scatter(
                x=daily_data.index,
                y=daily_data['event_count'],
                name='Actual',
                line=dict(color='blue')
            ))
            
            # 예측 데이터
            fig_forecast.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions.flatten(),
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig_forecast.update_layout(
                title='Daily Activity Forecast',
                xaxis_title='Date',
                yaxis_title='Number of Events',
                showlegend=True
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # 예측 결과 요약
            st.subheader("Forecast Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                avg_current = daily_data['event_count'].mean()
                avg_forecast = np.mean(future_predictions)
                growth_rate = ((avg_forecast - avg_current) / avg_current) * 100
                create_metric_card(
                    "Expected Growth Rate",
                    f"{growth_rate:.1f}%",
                    delta=f"{growth_rate:.1f}%",
                    delta_color="normal" if growth_rate > 0 else "inverse"
                )
            
            with col2:
                max_forecast = np.max(future_predictions)
                create_metric_card(
                    "Peak Activity Forecast",
                    f"{int(max_forecast)}",
                    delta=f"{int(max_forecast - avg_current)}",
                    delta_color="normal"
                )

    with tab6:
        st.header('System Architecture')
        
        # Data Flow Architecture
        st.subheader('Data Flow Architecture')
        data_flow_diagram = """
        graph TD
            A[Firebase Realtime DB] -->|Real-time Stream| B[Data Collector]
            A -->|Batch Load| B
            B -->|Processed Data| C[Data Storage]
            C -->|Analysis Ready| D[Analytics Engine]
            D -->|Insights| E[Dashboard]
            D -->|Predictions| E
            D -->|Visualizations| E
        """
        components.html(f"""
            <div class="mermaid">
            {data_flow_diagram}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true}});</script>
        """, height=400)
        
        # Storage Architecture
        st.subheader('Storage Architecture')
        storage_diagram = """
        graph LR
            A[Firebase Realtime DB] -->|Primary Storage| B[Data Layer]
            B -->|Cache| C[Local Cache]
            B -->|Process| D[Analysis Storage]
            D -->|Results| E[Visualization Cache]
        """
        components.html(f"""
            <div class="mermaid">
            {storage_diagram}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true}});</script>
        """, height=300)
        
        # ETL Process Flow
        st.subheader('ETL Process Flow')
        etl_diagram = """
        graph TD
            A[Extract] -->|Raw Data| B[Transform]
            B -->|Cleaned Data| C[Load]
            C -->|Processed Data| D[Analysis]
            D -->|Results| E[Visualization]
            
            subgraph Extract
            A1[Firebase Events] --> A
            A2[Firebase Projects] --> A
            A3[Firebase Portfolios] --> A
            end
            
            subgraph Transform
            B1[Data Cleaning] --> B
            B2[Feature Engineering] --> B
            B3[Data Normalization] --> B
            end
            
            subgraph Load
            C1[Quality Check] --> C
            C2[Data Validation] --> C
            C3[Storage Optimization] --> C
            end
        """
        components.html(f"""
            <div class="mermaid">
            {etl_diagram}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true}});</script>
        """, height=500)
        
        # Analytics Pipeline
        st.subheader('Analytics Pipeline')
        analytics_diagram = """
        graph TD
            A[Data Input] -->|Process| B[Analytics Engine]
            B -->|Descriptive| C[KPI Analysis]
            B -->|Diagnostic| D[Pattern Analysis]
            B -->|Predictive| E[ML Models]
            B -->|Prescriptive| F[Recommendations]
            
            C -->|Results| G[Dashboard]
            D -->|Results| G
            E -->|Results| G
            F -->|Results| G
        """
        components.html(f"""
            <div class="mermaid">
            {analytics_diagram}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true}});</script>
        """, height=400)
        
        # Component Architecture
        st.subheader('Component Architecture')
        component_diagram = """
        graph TD
            A[Dashboard.py] -->|Uses| B[DataCollector]
            A -->|Uses| C[DataAnalyzer]
            B -->|Collects| D[Firebase Data]
            C -->|Analyzes| E[Processed Data]
            A -->|Displays| F[Streamlit UI]
            F -->|Shows| G[Visualizations]
            F -->|Shows| H[Metrics]
            F -->|Shows| I[Predictions]
        """
        components.html(f"""
            <div class="mermaid">
            {component_diagram}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true}});</script>
        """, height=400)

if __name__ == "__main__":
    main() 