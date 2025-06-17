import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timedelta
import os
import json

class DataCollector:
    def __init__(self):
        # Check if Firebase is already initialized
        if not firebase_admin._apps:
            try:
                # Use service account key file
                cred = credentials.Certificate("src/service-account/service-account.json")
                firebase_admin.initialize_app(cred)
            except Exception as e:
                print(f"Error initializing Firebase: {str(e)}")
                raise
        
        self.db = firestore.client()
        self.cache_dir = 'data/cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 모든 컬렉션 목록 출력
        print("\n=== Available Collections ===")
        collections = self.db.collections()
        for collection in collections:
            print(f"Collection: {collection.id}")
            # 각 컬렉션의 첫 번째 문서 샘플 출력
            docs = collection.limit(1).stream()
            for doc in docs:
                print(f"Sample document data: {doc.to_dict()}")
        print("===========================\n")
    
    def collect_events(self, days=90):
        """이벤트 데이터 수집"""
        cache_file = f'{self.cache_dir}/events.csv'
        
        # 캐시된 데이터가 있으면 로드
        if os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age.days < 1:  # 1일 이내의 캐시면 사용
                print("Loading events from cache...")
                return pd.read_csv(cache_file)
        
        print("Collecting events from Firebase...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        print(f"Date range: {start_date} to {end_date}")
        
        try:
            # 모든 컬렉션에서 이벤트 데이터 찾기
            collections = self.db.collections()
            events_data = []
            
            for collection in collections:
                print(f"\nChecking collection: {collection.id}")
                # 컬렉션의 모든 문서 가져오기
                docs = collection.stream()
                for doc in docs:
                    data = doc.to_dict()
                    print(f"Document data: {data}")
                    
                    # 컬렉션 이름에 따라 이벤트 타입 결정
                    event_type = None
                    if collection.id == 'portfolios':
                        event_type = 'portfolio_created'
                        # 공유된 포트폴리오는 추가 이벤트로 기록
                        if data.get('isShared'):
                            events_data.append({
                                'id': f"{doc.id}_shared",
                                'user_id': data.get('userId'),
                                'event_type': 'portfolio_shared',
                                'timestamp': data.get('createdAt')
                            })
                    elif collection.id == 'projects':
                        event_type = 'project_created'
                    elif collection.id == 'ideas':
                        event_type = 'idea_created'
                    
                    if event_type:
                        event_data = {
                            'id': doc.id,
                            'user_id': data.get('userId'),
                            'event_type': event_type,
                            'timestamp': data.get('createdAt')
                        }
                        events_data.append(event_data)
                        print(f"Added event: {event_data}")
            
            print(f"\nTotal events found: {len(events_data)}")
            
            if not events_data:
                print("No events found in any collection")
                return pd.DataFrame()
            
            df = pd.DataFrame(events_data)
            print("DataFrame columns:", df.columns.tolist())
            
            # timestamp를 datetime으로 변환
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 데이터 캐시
            if not df.empty:
                df.to_csv(cache_file, index=False)
                print(f"Saved {len(df)} events to cache")
            
            return df
            
        except Exception as e:
            print(f"Error collecting events: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
    
    def collect_portfolio_data(self):
        """포트폴리오 데이터 수집"""
        cache_file = f'{self.cache_dir}/portfolios.csv'
        
        # 캐시된 데이터가 있으면 로드
        if os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age.days < 1:  # 1일 이내의 캐시면 사용
                print("Loading portfolios from cache...")
                return pd.read_csv(cache_file)
        
        print("Collecting portfolios from Firebase...")
        try:
            portfolios_ref = self.db.collection('portfolios')
            portfolios = portfolios_ref.stream()
            
            portfolio_data = []
            for portfolio in portfolios:
                data = portfolio.to_dict()
                data['id'] = portfolio.id
                # 컬럼명 변환
                if 'isShared' in data:
                    data['is_shared'] = data.pop('isShared')
                if 'projectIds' in data:
                    data['project_count'] = len(data.pop('projectIds'))
                portfolio_data.append(data)
            
            print(f"Collected {len(portfolio_data)} portfolios")
            
            if not portfolio_data:
                print("No portfolios found")
                return pd.DataFrame()
            
            df = pd.DataFrame(portfolio_data)
            print("DataFrame columns:", df.columns.tolist())
            
            # 데이터 캐시
            if not df.empty:
                df.to_csv(cache_file, index=False)
                print(f"Saved {len(df)} portfolios to cache")
            
            return df
            
        except Exception as e:
            print(f"Error collecting portfolios: {str(e)}")
            return pd.DataFrame()
    
    def collect_project_data(self):
        """프로젝트 데이터 수집"""
        cache_file = f'{self.cache_dir}/projects.csv'
        
        # 캐시된 데이터가 있으면 로드
        if os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age.days < 1:  # 1일 이내의 캐시면 사용
                print("Loading projects from cache...")
                return pd.read_csv(cache_file)
        
        print("Collecting projects from Firebase...")
        try:
            projects_ref = self.db.collection('projects')
            projects = projects_ref.stream()
            
            project_data = []
            for project in projects:
                data = project.to_dict()
                data['id'] = project.id
                project_data.append(data)
            
            print(f"Collected {len(project_data)} projects")
            
            if not project_data:
                print("No projects found")
                return pd.DataFrame()
            
            df = pd.DataFrame(project_data)
            print("DataFrame columns:", df.columns.tolist())
            
            # 데이터 캐시
            if not df.empty:
                df.to_csv(cache_file, index=False)
                print(f"Saved {len(df)} projects to cache")
            
            return df
            
        except Exception as e:
            print(f"Error collecting projects: {str(e)}")
            return pd.DataFrame()
    
    def save_to_csv(self, df, filename):
        """데이터프레임을 CSV 파일로 저장"""
        os.makedirs('data', exist_ok=True)
        df.to_csv(f'data/{filename}.csv', index=False) 