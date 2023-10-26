import matplotlib.pyplot as plt
import numpy as np

color_dict_compare = {'our': 'forestgreen', 'ref': '#2980b9', 'compare1': '#F3B234', 'compare2': '#DB8647', 'compare3': '#CA443B'}
#8D9D81
# mark_dict_compare = {'our':'.', 'compare1': '^', 'compare2':'+', 'compare3':'D'}
color_dict_compare_deep = {
    'ref': '#0b4070',         # dark blue
    'our': '#0d6e3f',         # dark green
    'compare1': '#b3740d',    # dark orange
    'compare2': '#a63e1e',    # dark red
    'compare3': '#84342a',    # dark brown
    'compare4': '#2b2b2b',    # dark gray
    'compare5': '#990099',    # dark purple
    'compare6': '#7e7e7e',    # dark silver
    'compare7': '#800000'     # dark maroon
}


color_dict_single = {'error': '#2980b9', 'para': '#d35400'}

def Parameter_interate(para_history, parameter_name, para_ref = None, line_width=2.5,font_size = 26, color = None):
    para_history = np.reshape(para_history, (1, -1))[0]
    if color == None:
        color = color_dict_single['para']
    if para_ref != None:
        plt.figure(dpi=100,figsize=(8,6))
        para_ref_history = para_ref * np.ones(len(para_history))
        plt.plot(range(len(para_history)), np.abs(para_history - para_ref_history), color = color, linewidth=line_width, label = 'error of parameter %s' % parameter_name)
        plt.xlabel('iterations',fontsize=font_size)
        plt.ylabel('error',fontsize=font_size)
        plt.yscale('log')
        plt.legend(fontsize=font_size)
        plt.tick_params(labelsize=20)

        # plt.savefig('./output/parameter %s error.png' % parameter_name)
        plt.show()
    else:
        plt.plot(range(len(para_history)), para_history, color = color, marker ='.')
        plt.xlabel('iterations')
        plt.ylabel('parameter %s ' % parameter_name)
        # plt.savefig('./output/parameter_%s.png' % parameter_name)
        plt.show()
    return
        
def error_interate(err_history, line_width=2.5, font_size = 26, color = None):
    if color == None:
        color = color_dict_single['error']
    plt.figure(dpi=100,figsize=(8,6))
    plt.plot(range(len(err_history)), err_history, color = color,linewidth=line_width, label = 'error of data')
    plt.xlabel('iterations',fontsize=font_size)
    plt.ylabel('error',fontsize=font_size)
    plt.yscale('log')
    plt.legend(fontsize=font_size)
    plt.tick_params(labelsize=font_size)

    plt.show()
    return