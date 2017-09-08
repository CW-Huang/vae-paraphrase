variables = {}
P_list = []


def track_variable(P, name, var):
    if P not in P_list:
        pid = len(P_list)
        P_list.append(P)
    else:
        pid = P_list.index(P)
    variables[pid, name] = var


def get_variable(P, name):
    for pid in xrange(len(P_list)):
        if P_list[pid] is P:
            break
    return variables[pid, name]
