def compare_numbers(num1, num2):
    return num1 == num2

# 输入两个数字字符串
number1 = input("请输入第一个数字字符串: ")
number2 = input("请输入第二个数字字符串: ")

# 比较两个数字字符串
if compare_numbers(number1, number2):
    print("两个数字字符串相同。")
else:
    print("两个数字字符串不同。")
