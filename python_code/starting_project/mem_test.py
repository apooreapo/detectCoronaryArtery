# import random
# import fft_transform
# import gc
# import sys
#
# def do_fourier(data):
#     # fourier_transform = fft_transform.FFTTransform(fs=100, data_series=data)
#     res9a, res9b, res28 = fft_transform.FFTTransform(fs=100, data_series=data).vlf_band()
#     # del fourier_transform
#     # gc.collect()
#     print(res9a)
#
# @profile
# def my_func(data, print_message=False):
#     res_dict = {}
#     column_titles = ["t1", "t2", "t3", "t4"]
#     for name in column_titles:
#         res_dict[name] = []
#         # do_fourier(my_list)
#     for jj in range(0, 2):
#         fourier_transform = fft_transform.FFTTransform(fs=100, data_series=data)
#         res10a, res10b, res29 = fourier_transform.lf_band()
#         res_dict[column_titles[0]].append(res10a)
#         res_dict[column_titles[1]].append(res10b)
#         if print_message:
#             print(f"LF band energy is {res10a}.")
#             print(f"LF band energy is {round(res10b * 100, 2)}% of the full energy.")
#             print(f"LF Energy peak is at {round(res29, 3)} Hz.")
#         res11a, res11b, res30 = fourier_transform.hf_band()
#         res_dict[column_titles[2]].append(res11a)
#         res_dict[column_titles[3]].append(res11b)
#         if print_message:
#             print(f"HF band energy is {res11a}.")
#             print(f"HF band energy is {round(res11b * 100, 2)}% of the full energy.")
#             print(f"HF Energy peak is at {round(res30, 3)} Hz.")
#         if res10a > 0:
#             res31 = res10a / res11a
#         else:
#             res31 = 0
#         if print_message:
#             print(f"LF / HF energy is {round(res31, 5)}.")
#         # print(res_dict)
#     return res_dict
#
# my_list = []
# for j in range(0, 10000):
#     k = 2 + j
#     my_list.append(k)
# new = my_func(data=my_list)
# print(new)

orestis = [3, 4, 5, 6, 7, 8, 9]

print(orestis[1:])
print(orestis[:-1])
