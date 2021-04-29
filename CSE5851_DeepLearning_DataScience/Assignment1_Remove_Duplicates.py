# Assignment 1
# 2020312086 Hong Gibong

# 3. Remove Duplicates

def rm_duplicates(lst: list):
    '''
    find unique elements in given list.

    :param lst: Input list.
    :return: unique values in new list.
    '''
    new_lst = [] # create new list to append unique values
    for i in lst:
        if i not in new_lst: # check duplicates
            new_lst.append(i)
    return new_lst

print(rm_duplicates([2,5,2,3,5,'s','s','a']))