import pandas as pd

columns = [
'duration','protocol_type','service','flag','src_bytes','dst_bytes',
'land','wrong_fragment','urgent','hot','num_failed_logins',
'logged_in','num_compromised','root_shell','su_attempted',
'num_root','num_file_creations','num_shells','num_access_files',
'num_outbound_cmds','is_host_login','is_guest_login','count',
'srv_count','serror_rate','srv_serror_rate','rerror_rate',
'srv_rerror_rate','same_srv_rate','diff_srv_rate',
'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
'dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
'dst_host_serror_rate','dst_host_srv_serror_rate',
'dst_host_rerror_rate','dst_host_srv_rerror_rate',
'label','difficulty'
]

# Load train dataset
train_df = pd.read_csv("KDDTrain+.txt", names=columns)
train_df.drop("difficulty", axis=1, inplace=True)
train_df.to_csv("dataset_train.csv", index=False)

# Load test dataset
test_df = pd.read_csv("KDDTest+.txt", names=columns)
test_df.drop("difficulty", axis=1, inplace=True)
test_df.to_csv("dataset_test.csv", index=False)

print("Conversion Successful ✅")
