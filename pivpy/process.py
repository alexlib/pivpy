""" Processing PIV flow fields """

def averf(data):
    """
    averf(d) creates ensemble average field
    """
    
    av = data[0]
    for d in data[1:]:
        av['u'] += d['u']
        av['v'] += d['v']
    
    av['u'] /= len(d)
    av['v'] /= len(d)
    
    return av