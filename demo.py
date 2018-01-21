from NLP.sentence_transform import sentence_transform

train_data = ['全面从严治党',
              '国际公约和国际法',
              '中国航天科技集团有限公司']
test_data = ['全面从严测试']
print('sentence_transform(train_data=train_data,hash=False)\n',
      sentence_transform(train_data=train_data, hash=False))
print('sentence_transform(train_data=train_data,hash=True)\n',
      sentence_transform(train_data=train_data, hash=True))
m, n = sentence_transform(train_data=train_data, test_data=test_data, hash=True)
print('sentence_transform(train_data=train_data,test_data=test_data,hash=True)\n',
      'train_data\n', m, '\ntest_data\n', n)
