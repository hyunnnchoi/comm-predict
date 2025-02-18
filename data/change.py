import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('arps_a100_8gpu_network_summary.csv')

# 연속적인 인덱스로 Job ID 열 업데이트
df['Job ID'] = range(len(df))

# 수정된 데이터를 CSV 파일로 저장
df.to_csv('updated_paste.txt', index=False)

# 결과 확인
print("처음 10개 행:")
print(df.head(10))