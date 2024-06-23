import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),dpi=300)

# 设置格子数量和边长
num_cells = 5
cell_size = 1.0

noinx = [0,0,0,1,1,3,3,3]
noiny = [0,1,3,0,3,2,1,4]
# 绘制左侧表格
ax1.set_xlim(0, (num_cells) * cell_size)
ax1.set_ylim(0, (num_cells + 2) * cell_size)
ax1.set_aspect('equal')  # 设置纵横比例为1:1
ax1.spines['top'].set_visible(False)  # 隐藏顶部边框
ax1.spines['right'].set_visible(False)  # 隐藏右侧边框
ax1.spines['bottom'].set_visible(False)  # 隐藏底部边框
ax1.spines['left'].set_visible(False)  # 隐藏左侧边框
for i in range(num_cells):
    for j in range(num_cells-1):
        # 计算格子左下角坐标
        x = j * cell_size
        y = i * cell_size

        # 绘制格子
        rect = patches.Rectangle((x, y), cell_size, cell_size, linewidth=1, edgecolor='black', facecolor='none')
        ax1.add_patch(rect)

        # 计算内切圆的半径和圆心坐标
        radius = cell_size / 2
        center_x = x + radius
        center_y = y + radius

        # 绘制内切圆，并设置填充颜色为红色
        circle = patches.Circle((center_x, center_y), radius, linewidth=1, edgecolor='black', facecolor='white')
        ax1.add_patch(circle)

for i in range(len(noinx)):
        x = noinx[i] * cell_size
        y = noiny[i] * cell_size
        radius = cell_size / 2
        center_x = x + radius
        center_y = y + radius

        circle = patches.Circle((center_x, center_y), radius, linewidth=1, edgecolor='black', facecolor='#23238E')
        ax1.add_patch(circle)
rect = patches.Rectangle((2.5, 5.5), cell_size, cell_size, linewidth=1, edgecolor='black', facecolor='none')
ax1.add_patch(rect)
circle = patches.Circle((3, 6), 0.5, linewidth=1, edgecolor='black', facecolor='#23238E')
ax1.add_patch(circle)

rect = patches.Rectangle((0.5, 5.5), cell_size, cell_size, linewidth=1, edgecolor='black', facecolor='none')
ax1.add_patch(rect)
circle = patches.Circle((1, 6), 0.5, linewidth=1, edgecolor='black', facecolor='white')
ax1.add_patch(circle)


ax1.set_xticks([])
ax1.set_yticks([])
ax1.text(2, -0.2, 'Time', ha='center', va='center',fontproperties={'family':'Times New Roman'})
ax1.text(-0.2, 2.5, 'Location', ha='center', va='center', rotation=90,fontproperties={'family':'Times New Roman'})
ax1.text(1, 6.8, 'Empty', ha='center', va='center',fontproperties={'family':'Times New Roman'})
ax1.text(3, 6.8, 'Sampled', ha='center', va='center',fontproperties={'family':'Times New Roman'})
ax2.text(1, 6.8, 'Inferred', ha='center', va='center',fontproperties={'family':'Times New Roman'})
ax2.text(3, 6.8, 'Sampled', ha='center', va='center',fontproperties={'family':'Times New Roman'})
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Location')
# 绘制右侧表格，代码与左侧表格类似，只需修改坐标轴和标题
for i in range(num_cells):
    for j in range(num_cells-1):
        # 计算格子左下角坐标
        x = j * cell_size
        y = i * cell_size

        # 绘制格子
        rect = patches.Rectangle((x, y), cell_size, cell_size, linewidth=1, edgecolor='black', facecolor='none')
        ax2.add_patch(rect)

        # 计算内切圆的半径和圆心坐标
        radius = cell_size / 2
        center_x = x + radius
        center_y = y + radius

        # 绘制内切圆，并设置填充颜色为蓝色
        circle = patches.Circle((center_x, center_y), radius, linewidth=1, edgecolor='black', facecolor='#DB70DB')
        ax2.add_patch(circle)
for i in range(len(noinx)):
        x = noinx[i] * cell_size
        y = noiny[i] * cell_size
        radius = cell_size / 2
        center_x = x + radius
        center_y = y + radius

        # 绘制内切圆，并设置填充颜色为红色
        circle = patches.Circle((center_x, center_y), radius, linewidth=1, edgecolor='black', facecolor='#23238E')
        ax2.add_patch(circle)

rect = patches.Rectangle((0.5, 5.5), cell_size, cell_size, linewidth=1, edgecolor='black', facecolor='none')
ax2.add_patch(rect)
circle = patches.Circle((1, 6), 0.5, linewidth=1, edgecolor='black', facecolor='#DB70DB')
ax2.add_patch(circle)

rect = patches.Rectangle((2.5, 5.5), cell_size, cell_size, linewidth=1, edgecolor='black', facecolor='none')
ax2.add_patch(rect)
circle = patches.Circle((3, 6), 0.5, linewidth=1, hatch='//',edgecolor='black', facecolor='#23238E')
ax2.add_patch(circle)
# 设置右侧坐标轴范围和纵横比例
ax2.set_xlim(0, (num_cells) * cell_size)
ax2.set_ylim(0, (num_cells+2) * cell_size)
ax2.set_aspect('equal')  # 设置纵横比例为1:1
ax2.spines['top'].set_visible(False)  # 隐藏顶部边框
ax2.spines['right'].set_visible(False)  # 隐藏右侧边框
ax2.spines['bottom'].set_visible(False)  # 隐藏底部边框
ax2.spines['left'].set_visible(False)  # 隐藏左侧边框
# ax2.set_title('Right Table')  # 设置右侧表格标题
ax2.set_xticks([])
ax2.set_yticks([])
ax2.text(2, -0.2, 'Time', ha='center', va='center',fontproperties={'family':'Times New Roman'})
ax2.text(-0.2, 2.5, 'Location', ha='center', va='center', rotation=90,fontproperties={'family':'Times New Roman'})
# 隐藏坐标轴
# ax1.axis('off')
# ax2.axis('off')

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.2)

# 显示图形
plt.show()
# plt.savefig('4-1-inferred.svg')