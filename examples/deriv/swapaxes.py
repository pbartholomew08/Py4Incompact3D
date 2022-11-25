import numpy as np

def is_view(a):
    return not a.base is None

nx = 3
ny = 5

a = np.zeros((nx, ny))


for i in range(nx):
    for j in range(ny):
        a[i,j] = i * ny + j

b = np.swapaxes(a, 0, 1)
print(is_view(b))
c = np.copy(a)
print(is_view(c))
d = np.copy(b)
e = np.copy(np.swapaxes(b, 0, 1))

print(a)
print(b)

print(np.flip(a, axis=-1))
print(np.flip(b))

#print(f"a: {is_view(a)} {a.flags}")
#print(a)
#print("-"*72)
#print(f"b: {is_view(b)}")
#print(b)
#print("-"*72)
#print(f"c: {is_view(c)}")
#print(c)
#print("-"*72)
#print(f"d: {is_view(d)}")
#print(d)
#print("-"*72)
#print(f"e: {is_view(e)}")
#print(e)
#print("-"*72)
#
#print(f"{a[1,2]} {a[1][2]}")
#print(f"{b[1,2]} {b[1][2]}")
#print(f"{a[:,2]}")
#print(f"{b[:,2]}")
