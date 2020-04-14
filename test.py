if __name__ == '__main__':
    list_n = []
    for i in range(1,50):
        if (i%5)==0:
            list_n.append([str('{:03d}').format(i),5/i])
    print('list', list)
    list_n.append(['07899', 890098])
    for i in range(len(list_n)):
        print(list_n[i], "{:03d}".format(i) ,end=' ')
    print(' ')
    m = max(list_n, key=lambda x: x[1])
    print(m[0])