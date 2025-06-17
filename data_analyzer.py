import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import networkx as nx
from collections import defaultdict

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        if 'timestamp' in df.columns:
            self.df['date'] = pd.to_datetime(self.df['timestamp'])
    
    def get_event_trends(self):
        """이벤트 추이 분석"""
        return self.df.groupby(['date', 'event_name']).size().reset_index(name='count')
    
    def get_funnel_data(self):
        """전환 퍼널 분석"""
        funnel_steps = ['page_loaded', 'idea_created', 'project_created', 'portfolio_submitted']
        funnel_data = []
        for step in funnel_steps:
            count = len(self.df[self.df['event_name'] == step])
            funnel_data.append({'step': step, 'count': count})
        return pd.DataFrame(funnel_data)
    
    def get_bounce_rate(self):
        """이탈률 계산 (베이지안 업데이트)"""
        if self.df.empty:
            return 0.0
        
        # 한 번의 이벤트만 발생한 사용자 수
        single_event_users = self.df.groupby('user_id').filter(lambda x: len(x) == 1)
        bounce_users = len(single_event_users['user_id'].unique())
        total_users = len(self.df['user_id'].unique())
        
        return bounce_users / total_users if total_users > 0 else 0.0
    
    def get_user_journeys(self):
        """사용자 여정 분석"""
        user_journeys = self.df.sort_values('timestamp').groupby('user_id')['event_name'].agg(list)
        return user_journeys.value_counts().head(10)
    
    def get_hourly_activity(self):
        """시간대별 활동량 분석"""
        self.df['hour'] = self.df['date'].dt.hour
        return self.df.groupby('hour').size().reset_index(name='count')
    
    def get_portfolio_stats(self):
        """포트폴리오 통계"""
        if self.df.empty:
            return None
        
        # 포트폴리오 관련 이벤트 필터링
        portfolio_events = self.df[self.df['event_type'].str.contains('portfolio')]
        
        if portfolio_events.empty:
            return None
        
        stats = {
            'avg_sections': portfolio_events['sections'].mean() if 'sections' in portfolio_events.columns else 0,
            'avg_projects': portfolio_events['project_count'].mean() if 'project_count' in portfolio_events.columns else 0,
            'shared_rate': len(portfolio_events[portfolio_events['event_type'] == 'portfolio_shared']) / len(portfolio_events) * 100
        }
        
        return stats
    
    def get_conversion_rate(self):
        """전환율 계산 (조건부 확률)"""
        if self.df.empty:
            return 0.0
        
        # 아이디어 생성 후 프로젝트 생성한 사용자 수
        idea_users = set(self.df[self.df['event_type'] == 'idea_created']['user_id'])
        project_users = set(self.df[self.df['event_type'] == 'project_created']['user_id'])
        
        converted_users = len(idea_users.intersection(project_users))
        return converted_users / len(idea_users) if len(idea_users) > 0 else 0.0
    
    def get_completion_rate(self):
        """프로젝트 완료율 계산 (베르누이 분포)"""
        if self.df.empty:
            return 0.0
        
        # 프로젝트 생성 후 완료한 사용자 수
        project_users = set(self.df[self.df['event_type'] == 'project_created']['user_id'])
        completed_users = set(self.df[self.df['event_type'] == 'project_completed']['user_id'])
        
        return len(completed_users) / len(project_users) if len(project_users) > 0 else 0.0
    
    def get_idea_to_project_rate(self):
        """아이디어-프로젝트 전환율 계산"""
        if self.df.empty:
            return 0.0
        
        total_ideas = len(self.df[self.df['event_type'] == 'idea_created'])
        total_projects = len(self.df[self.df['event_type'] == 'project_created'])
        
        return total_projects / total_ideas if total_ideas > 0 else 0.0
    
    def get_project_portfolio_graph(self):
        """프로젝트-포트폴리오 관계 그래프 데이터 생성"""
        if self.df.empty:
            return None
        
        # 프로젝트와 포트폴리오 이벤트 추출
        project_events = self.df[self.df['event_type'] == 'project_created']
        portfolio_events = self.df[self.df['event_type'] == 'portfolio_submitted']
        
        if project_events.empty or portfolio_events.empty:
            return None
        
        # 그래프 생성
        G = nx.Graph()
        
        # 노드 추가
        for _, row in project_events.iterrows():
            G.add_node(f"P_{row['user_id']}", type='project')
        
        for _, row in portfolio_events.iterrows():
            G.add_node(f"PF_{row['user_id']}", type='portfolio')
        
        # 엣지 추가
        for _, row in project_events.iterrows():
            G.add_edge(f"P_{row['user_id']}", f"PF_{row['user_id']}")
        
        # 레이아웃 계산
        pos = nx.spring_layout(G)
        
        # 그래프 데이터 생성
        graph_data = {
            'x': [pos[node][0] for node in G.nodes()],
            'y': [pos[node][1] for node in G.nodes()],
            'labels': [node for node in G.nodes()]
        }
        
        return graph_data
    
    def get_network_metrics(self):
        """네트워크 메트릭스 계산"""
        if self.df.empty:
            return {
                'avg_degree': 0.0,
                'clustering_coef': 0.0,
                'density': 0.0
            }
        
        # 프로젝트와 포트폴리오 이벤트 추출
        project_events = self.df[self.df['event_type'] == 'project_created']
        portfolio_events = self.df[self.df['event_type'] == 'portfolio_submitted']
        
        if project_events.empty or portfolio_events.empty:
            return {
                'avg_degree': 0.0,
                'clustering_coef': 0.0,
                'density': 0.0
            }
        
        # 그래프 생성
        G = nx.Graph()
        
        # 노드 추가
        for _, row in project_events.iterrows():
            G.add_node(f"P_{row['user_id']}", type='project')
        
        for _, row in portfolio_events.iterrows():
            G.add_node(f"PF_{row['user_id']}", type='portfolio')
        
        # 엣지 추가
        for _, row in project_events.iterrows():
            G.add_edge(f"P_{row['user_id']}", f"PF_{row['user_id']}")
        
        # 메트릭스 계산
        metrics = {
            'avg_degree': sum(dict(G.degree()).values()) / len(G),
            'clustering_coef': nx.average_clustering(G),
            'density': nx.density(G)
        }
        
        return metrics
    
    def get_funnel_data(self, steps):
        """전환 퍼널 데이터 계산"""
        if self.df.empty:
            return pd.Series()
        
        funnel_data = {}
        for step in steps:
            count = len(self.df[self.df['event_type'] == step])
            funnel_data[step] = count
        
        return pd.Series(funnel_data)
    
    def get_user_journeys(self):
        """사용자 행동 흐름 데이터 생성"""
        if self.df.empty:
            return pd.Series()
        
        # 사용자별 이벤트 시퀀스 생성
        user_journeys = self.df.sort_values('timestamp').groupby('user_id')['event_type'].agg(list)
        
        # 시퀀스를 문자열로 변환
        journey_strings = user_journeys.apply(lambda x: ' -> '.join(x))
        
        return journey_strings
    
    def get_event_sequences(self):
        """이벤트 시퀀스 분석"""
        if self.df.empty:
            return pd.DataFrame()
        
        # 사용자별 이벤트 시퀀스 생성
        sequences = []
        for user_id in self.df['user_id'].unique():
            user_events = self.df[self.df['user_id'] == user_id].sort_values('timestamp')
            if len(user_events) >= 2:  # 최소 2개 이상의 이벤트가 있는 경우만
                sequence = ' -> '.join(user_events['event_type'].values)
                sequences.append(sequence)
        
        # 시퀀스 빈도 계산
        sequence_counts = pd.Series(sequences).value_counts().reset_index()
        sequence_counts.columns = ['sequence', 'count']
        return sequence_counts
    
    def get_daily_activity(self):
        """일별 활동량 분석"""
        if self.df.empty:
            return pd.DataFrame()
        
        daily_activity = self.df.groupby(self.df['date'].dt.date).size().reset_index(name='count')
        daily_activity.columns = ['date', 'count']
        return daily_activity
    
    def get_user_segments(self):
        """사용자 세그먼트 분석"""
        if self.df.empty:
            return pd.DataFrame()
        
        # 사용자별 활동량 계산
        user_activity = self.df.groupby('user_id').size().reset_index(name='activity_count')
        
        # 활동량 기준으로 4분위수로 세그먼트 분류
        user_activity['segment'] = pd.qcut(
            user_activity['activity_count'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return user_activity
    
    def get_conversion_trends(self):
        """전환율 추이 분석"""
        if self.df.empty:
            return pd.DataFrame()
        
        # 일별 전환율 계산
        daily_conversion = self.df.groupby(self.df['date'].dt.date).apply(
            lambda x: len(x[x['event_type'] == 'project_created']) / len(x[x['event_type'] == 'idea_created'])
            if len(x[x['event_type'] == 'idea_created']) > 0 else 0
        ).reset_index(name='conversion_rate')
        
        daily_conversion.columns = ['date', 'conversion_rate']
        return daily_conversion 