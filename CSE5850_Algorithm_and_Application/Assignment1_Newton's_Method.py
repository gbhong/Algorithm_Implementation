# Assignment 1_Algorithm and App.
# 2020312086 Hong Gibong

# 4. Newton's Method

def squareRoot(num, ans, tol):
    ''' Function to obtain square roots recursively '''
    if abs(ans**2-num) <= tol:
        return ans
    else:
        return squareRoot(num, (ans**2+num)/(2*ans), tol)

print(squareRoot(2.0, 2.0, 1e-2))