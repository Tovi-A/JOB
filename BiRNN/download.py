#下载用于训练和测试的mnist数据集的源码
 
import input_data # 调用input_data
print('Download and Extract MNISIT dataset:  ')
mnist = input_data.read_data_sets('data/', one_hot=True)
 
print("type of 'mnnist' is %s"% (type(mnist)))
print("number of train data is %d " % (mnist.train.num_examples))
print("number of test data is %d" %(mnist.test.num_examples))

