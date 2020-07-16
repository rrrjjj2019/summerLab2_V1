import tensorflow as tf

# Pythonic
lucky_number = 24
print("Pythonic:")
print(lucky_number)
print("\n")

# TensorFlow
lucky_number = tf.Variable(24, name = "lucky_number")
lucky_number2 = tf.Variable(24, name = "lucky_number2")
print("TensorFlow:")
with tf.Session() as sess:
  sess.run(lucky_number.initializer)
  sess.run(lucky_number2.initializer)
  temp = tf.add(lucky_number, lucky_number2)
  temp2 = tf.add(temp, lucky_number2)
  #print(sess.run(lucky_number))
  #print("########################## {} ".format(temp))
  #print("########################## {} ".format(sess.run(temp)))
  print("########################## {} ".format(temp2))
  print("########################## {} ".format(sess.run(temp2)))