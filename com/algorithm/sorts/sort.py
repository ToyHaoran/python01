
from __future__ import print_function


def 冒泡排序():
    if 0:
        # https://en.wikipedia.org/wiki/Bubble_sort
        # 看图明白的快。

        def bubble_sort(collection):
            length = len(collection)
            for i in range(length):
                swapped = False
                for j in range(length-1):
                    if collection[j] > collection[j+1]:
                        swapped = True
                        collection[j], collection[j+1] = collection[j+1], collection[j]
                if not swapped: break
            return collection

        print(bubble_sort([0, 5, 3, 2, 2]))
        print(bubble_sort([]))
        print(bubble_sort([-2, -5, -45]))
        print(bubble_sort([27, 33, 28, 4, 2, 26, 13, 35, 8, 14]))

def 桶排序():
    if 1:
        # https://en.wikipedia.org/wiki/Bucket_sort

        import math

        def bucket_sort(mylist, bucket_size=5):
            if len(mylist) == 0:
                print('You don\'t have any elements in array!')

            min_value = mylist[0]
            max_value = mylist[0]

            # For finding minimum and maximum values
            for i in range(0, len(mylist)):
                if mylist[i] < min_value:
                    min_value = mylist[i]
                elif mylist[i] > max_value:
                    max_value = mylist[i]

            # Initialize buckets
            bucketCount = math.floor((max_value - min_value) / bucket_size) + 1
            buckets = []
            for i in range(0, bucketCount):
                buckets.append([])

            # For putting values in buckets
            for i in range(0, len(mylist)):
                buckets[math.floor((mylist[i] - min_value) / bucket_size)].append(mylist[i])

            # Sort buckets and place back into input array
            sortedArray = []
            for i in range(0, len(buckets)):
                insertion_sort(buckets[i])
                for j in range(0, len(buckets[i])):
                    sortedArray.append(buckets[i][j])

            return sortedArray


        print(bucket_sort([12, 23, 4, 5, 3, 2, 12, 81, 56, 95]))

if __name__ == '__main__':
    冒泡排序()
    桶排序()
