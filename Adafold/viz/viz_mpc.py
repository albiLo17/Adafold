import matplotlib.pyplot as plt
import numpy as np
def plot_vectors_3d(vectors=None,
                    action=None,
                    points=None,
                    predicted_points=None,
                    gripper_pos=None,
                    title=None):
    fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    axes = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
    titles = ['top', 'front', 'lateral f', 'lateral b']
    views = [(90, 0), (0, 0), (30, -45), (30, -105)]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    for n, ax in enumerate(axes):

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='grey', alpha=0.3, s=2, label='State')
        if predicted_points is not None:
            ax.scatter(predicted_points[:, 0], predicted_points[:, 1], predicted_points[:, 2], alpha=0.1, s=2,
                       color='b',
                       label='Next state')

        if action is not None:
            if gripper_pos is None:
                start_point = points[0][6:9]
            else:
                start_point = gripper_pos
            for i, a in enumerate(action):
                final_point = [start_point[0] + a[0], start_point[1] + a[1], start_point[2] + a[2]]
                ax.quiver(start_point[0], start_point[1], start_point[2],
                          final_point[0] - start_point[0], final_point[1] - start_point[1], final_point[2] - start_point[2],
                          color='r', linewidth=2, label=f'Action {i}')
                start_point = final_point

        ax.set_xlim([-0.15, 0.15])
        ax.set_ylim([-0.15, 0.15])
        # ax.invert_yaxis()
        ax.set_zlim([-0.1, 0.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(views[n][0], views[n][1])
        ax.set_title(titles[n])
        ax.set_axis_off()
        ax.set_facecolor('white')
        # plt.legend()

    if title is not None:
        plt.title(title)
    plt.tight_layout()
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    # plt.show()
    return fig


def plot_prediction_and_state(points=None, predicted_points=None, title=None):
    fig, axs = plt.subplots(3, 2, subplot_kw={'projection': '3d'})
    axes = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
    titles = ['top', 'front', 'lateral f']
    views = [(90, 0), (0, 0), (30, -45)]

    for n in range(3):

        axs[n][0].scatter(points[:, 0], points[:, 1], points[:, 2], color='grey', alpha=0.3, s=2,
                          label='Real Final State')
        if predicted_points is not None:
            axs[n][1].scatter(predicted_points[:, 0], predicted_points[:, 1], predicted_points[:, 2], alpha=0.1, s=2,
                              color='b',
                              label='Model Prediction')

        for ax in [axs[n][0], axs[n][1]]:
            ax.set_xlim([-0.15, 0.15])
            ax.set_ylim([-0.15, 0.15])
            # ax.invert_yaxis()
            ax.set_zlim([-0.1, 0.2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(views[n][0], views[n][1])
            ax.set_title(titles[n])
            ax.set_axis_off()
            ax.set_facecolor('white')
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    handles_dict = {}
    for ax in axs.flatten():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in handles_dict:
                handles_dict[label] = handle

    unique_labels = list(handles_dict.keys())
    unique_handles = list(handles_dict.values())
    fig.legend(unique_handles, unique_labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(unique_labels))
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig


def plot_prediction_and_state_singleview(points=None, predicted_points=None, title=None, view='top'):
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    # axes = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
    titles = ['top', 'front', 'lateral f', 'lateral b']
    views = [(90, 0), (0, 0), (30, -45)]

    if view == 'top':
        n = 0
    elif view == 'front':
        n = 1
    else:
        if view == 'lateral b':
            views[2] = (30, 45)
            print(f'0: {views[2][0]}, 1: {views[2][1]}')
        n = 2


    axs[0].scatter(points[:, 0], points[:, 1], points[:, 2], color='grey', alpha=0.3, s=2,
                      label='Real Final State')
    if predicted_points is not None:
        axs[1].scatter(predicted_points[:, 0], predicted_points[:, 1], predicted_points[:, 2], alpha=0.1, s=2,
                          color='b',
                          label='Model Prediction')

    for ax in [axs[0], axs[1]]:
        ax.set_xlim([-0.15, 0.15])
        ax.set_ylim([-0.15, 0.15])
        # ax.invert_yaxis()
        ax.set_zlim([-0.1, 0.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(views[n][0], views[n][1])
        ax.set_title(titles[n])
        # ax.set_axis_off()
        ax.set_facecolor('white')
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    handles_dict = {}
    for ax in axs.flatten():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in handles_dict:
                handles_dict[label] = handle

    unique_labels = list(handles_dict.keys())
    unique_handles = list(handles_dict.values())
    fig.legend(unique_handles, unique_labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(unique_labels))
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig


def plot_state_singleview(points=None, full_points=None, full_points2=None, title=None, view='top'):
    fig = plt.figure(figsize=(12, 7))
    axs = fig.add_subplot(projection='3d')
    # fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    # axes = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
    titles = ['top', 'front', 'lateral f']
    views = [(90, 0), (0, 0), (30, -45)]
    if view == 'top':
        n = 0
    elif view == 'front':
        n = 1
    else:
        if view == 'lateral b':
            views[2] = (30, 225)
            print(f'0: {views[2][0]}, 1: {views[2][1]}')
        n = 2
    axs.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.5, s=30,
                label='Final State')
    if full_points is not None:
        axs.scatter(full_points[:, 0], full_points[:, 1], full_points[:, 2], color='grey', alpha=0.3, s=30,
                    label='bottom State')

    for ax in [axs]:
        ax.set_xlim([-0.15, 0.15])
        ax.set_ylim([-0.15, 0.15])
        # ax.invert_yaxis()
        ax.set_zlim([-0.1, 0.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(views[n][0], views[n][1])
        ax.set_title(titles[n])
        ax.set_axis_off()
        ax.set_facecolor('white')
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    handles_dict = {}
    handles, labels = axs.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in handles_dict:
            handles_dict[label] = handle
    unique_labels = list(handles_dict.keys())
    unique_handles = list(handles_dict.values())
    fig.legend(unique_handles, unique_labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(unique_labels))
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig


def plot_state_rec_singleview(points=None, full_points=None, full_points2=None, title=None, view='top'):
    fig = plt.figure(figsize=(12, 7))
    axs = fig.add_subplot(projection='3d')
    # fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    # axes = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
    titles = ['top', 'front', 'lateral f']
    views = [(90, 0), (0, 0), (30, -45)]
    if view == 'top':
        n = 0
    elif view == 'front':
        n = 1
    else:
        if view == 'lateral b':
            views[2] = (30, 225)
            print(f'0: {views[2][0]}, 1: {views[2][1]}')
        n = 2
    axs.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.5, s=30,
                label='State')
    if full_points is not None:
        axs.scatter(full_points[:, 0], full_points[:, 1], full_points[:, 2], color='grey', alpha=0.3, s=30,
                    label='Reconstruction')

    for ax in [axs]:
        ax.set_xlim([-0.15, 0.15])
        ax.set_ylim([-0.15, 0.15])
        # ax.invert_yaxis()
        ax.set_zlim([-0.1, 0.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(views[n][0], views[n][1])
        ax.set_title(titles[n])
        ax.set_axis_off()
        ax.set_facecolor('white')
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    handles_dict = {}
    handles, labels = axs.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in handles_dict:
            handles_dict[label] = handle
    unique_labels = list(handles_dict.keys())
    unique_handles = list(handles_dict.values())
    fig.legend(unique_handles, unique_labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(unique_labels))
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig


def plot_masked_state_singleview(points=None, mask1=None, mask2=None, title=None, view='top'):
    fig = plt.figure(figsize=(12, 7))
    axs = fig.add_subplot(projection='3d')
    # fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    # axes = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
    titles = ['top', 'front', 'lateral f']
    views = [(90, 0), (0, 0), (30, -45)]
    if view == 'top':
        n = 0
    elif view == 'front':
        n = 1
    else:
        if view == 'lateral b':
            views[2] = (30, 225)
            print(f'0: {views[2][0]}, 1: {views[2][1]}')
        n = 2
    axs.scatter(points[:, 0], points[:, 1], points[:, 2], color='grey', alpha=0.5, s=30,
                label='Full State')
    if mask1 is not None:
        axs.scatter(mask1[:, 0], mask1[:, 1], mask1[:, 2], color='tab:blue', alpha=0.3, s=30,
                    label='Mask GT State')
    if mask2 is not None:
        axs.scatter(mask2[:, 0], mask2[:, 1], mask2[:, 2], color='tab:orange', alpha=0.3, s=30,
                    label='Mask Pred State')


    for ax in [axs]:
        ax.set_xlim([-0.15, 0.15])
        ax.set_ylim([-0.15, 0.15])
        # ax.invert_yaxis()
        ax.set_zlim([-0.1, 0.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(views[n][0], views[n][1])
        ax.set_title(titles[n])
        ax.set_axis_off()
        ax.set_facecolor('white')
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    handles_dict = {}
    handles, labels = axs.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in handles_dict:
            handles_dict[label] = handle
    unique_labels = list(handles_dict.keys())
    unique_handles = list(handles_dict.values())
    fig.legend(unique_handles, unique_labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(unique_labels))
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig


def plot_state_multiview(points=None, predicted_points=None, title=None):
    fig, axs = plt.subplots(3, subplot_kw={'projection': '3d'})
    titles = ['top', 'front', 'lateral f']
    views = [(90, 0), (0, -90), (30, -45)]
    for n in range(3):
        axs[n].scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.2, s=0.5,
                       label='State')
        for ax in [axs[n]]:
            ax.set_xlim([-0.15, 0.15])
            ax.set_ylim([-0.15, 0.15])
            # ax.invert_yaxis()
            ax.set_zlim([-0.1, 0.2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(views[n][0], views[n][1])
            ax.set_title(titles[n])
            ax.set_axis_off()
            ax.set_facecolor('white')
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    handles_dict = {}
    for ax in axs.flatten():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in handles_dict:
                handles_dict[label] = handle
    unique_labels = list(handles_dict.keys())
    unique_handles = list(handles_dict.values())
    fig.legend(unique_handles, unique_labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(unique_labels))
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig




def plot_action_selection(vectors=None, action=None, points=None, full_points=None, start_point=None, use_vectors=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if start_point is None:
        gripper_pos = points[-1]
    else:
        gripper_pos = start_point

    if vectors is not None:
        for v in vectors:
            start_point = gripper_pos
            for i, a in enumerate(v):
                final_point = [start_point[0] + a[0], start_point[1] + a[1], start_point[2] + a[2]]
                ax.quiver(start_point[0], start_point[1], start_point[2],
                          final_point[0]-start_point[0], final_point[1]-start_point[1], final_point[2]-start_point[2], color='grey', alpha=0.7,  linewidth=1, label=f'Candidates')
                start_point = final_point

    start_point = gripper_pos
    for i, a in enumerate(action):
        final_point = [start_point[0] + a[0], start_point[1] + a[1], start_point[2] + a[2]]
        ax.quiver(start_point[0], start_point[1], start_point[2],
                  final_point[0]-start_point[0], final_point[1]-start_point[1], final_point[2]-start_point[2], color='r', linewidth=2, label=f'Action')
        start_point = final_point

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', alpha=0.9, s=3, label='State')
    if full_points is not None:
        ax.scatter(full_points[:, 0], full_points[:, 1], full_points[:, 2], color='grey', alpha=0.5, s=3, label='State')
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    # ax.invert_yaxis()
    ax.set_zlim([-0.1, 0.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Selected action')
    ax.view_init(30, -35)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    return fig


def plot_action_selection_with_values(vectors=None, action=None, points=None, full_points=None, start_point=None, Q_values=None, use_vectors=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if start_point is None:
        gripper_pos = points[-1]
    else:
        gripper_pos = start_point

    if vectors is not None:
        Q_min = min(Q_values)
        Q_max = max(Q_values)
        Q_range = Q_max - Q_min
        for v, q in zip(vectors, Q_values):
            normalized_Q = (q - Q_min) / Q_range if Q_range != 0 else 0.5
            color = plt.cm.viridis(normalized_Q)  # viridis colormap ranges from blue to yellow
            start_point = gripper_pos
            for i, a in enumerate(v):
                final_point = [start_point[0] + a[0], start_point[1] + a[1], start_point[2] + a[2]]
                ax.quiver(start_point[0], start_point[1], start_point[2],
                          a[0], a[1], a[2], color=color, alpha=0.7, linewidth=1)

    start_point = gripper_pos
    for i, a in enumerate(action):
        final_point = [start_point[0] + a[0], start_point[1] + a[1], start_point[2] + a[2]]
        ax.quiver(start_point[0], start_point[1], start_point[2],
                  a[0], a[1], a[2], color='r', linewidth=2, label=f'Action')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', alpha=0.9, s=3, label='State')
    if full_points is not None:
        ax.scatter(full_points[:, 0], full_points[:, 1], full_points[:, 2], color='grey', alpha=0.5, s=3,
                   label='Full State')
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([-0.1, 0.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Selected action')
    ax.view_init(30, -35)

    # Simplify the legend to only include unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    plt.legend(unique_handles, unique_labels)

    return fig



def plot_action_selection_2d(vectors=None, action=None, points=None, Q_values=None, grid_size=(30, 30)):
    fig, ax = plt.subplots()
    # Set up the grid
    grid_x, grid_y = np.meshgrid(np.linspace(-0.15, 0.15, grid_size[0]), np.linspace(-0.15, 0.15, grid_size[1]))
    grid_Q = np.full(grid_size, np.min(Q_values))  # Initialize with the minimum Q value
    if vectors is not None:
        for v, q in zip(vectors, Q_values):
            pos = np.array([0.0, 0.0])  # start at the origin, ensure it's float
            for a in v:
                pos += np.array(a[:2])  # only consider x, y
                # Find the closest grid cell to this vector's endpoint
                x_idx = np.searchsorted(np.linspace(-0.15, 0.15, grid_size[0]), pos[0], side='right') - 1
                y_idx = np.searchsorted(np.linspace(-0.15, 0.15, grid_size[1]), pos[1], side='right') - 1
                # Update the grid Q value if this vector's Q value is higher
                if 0 <= x_idx < grid_size[0] and 0 <= y_idx < grid_size[1]:
                    grid_Q[x_idx, y_idx] = np.maximum(grid_Q[x_idx, y_idx], q)
    # Normalize the Q values for color mapping
    Q_min = np.min(grid_Q)
    Q_max = np.max(grid_Q)
    norm = plt.Normalize(Q_min, Q_max)
    # Create a colormap for blue to yellow and map the normalized Q values to colors
    cmap = plt.cm.viridis
    colors = cmap(norm(grid_Q.ravel()))
    colors = colors.reshape(grid_x.shape + (4,))
    # Plot the grid cells with the average color
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # ax.add_patch(plt.Rectangle((grid_x[i, j] - 0.15 / grid_size[0], grid_y[i, j] - 0.15 / grid_size[1]),
            #                            0.15 * 2 / grid_size[0], 0.15 * 2 / grid_size[1],
            #                            color=colors[i, j], ec=None))
            ax.add_patch(plt.Rectangle((grid_x[i, j] - 0.15 / grid_size[0], grid_y[i, j] - 0.15 / grid_size[1]),
                                       0.15 * 2 / grid_size[0], 0.15 * 2/ grid_size[1],
                                       color=colors[i, j], ec=None))
    # Plot the action as a red arrow
    start_point = np.array([0.0, 0.0])  # start at the origin, ensure it's float
    for a in action:
        final_point = start_point + np.array(a[:2])  # only consider x, y
        ax.arrow(start_point[0], start_point[1],
                 final_point[0] - start_point[0], final_point[1] - start_point[1],
                 color='r', linewidth=2, head_width=0.01)
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], color='black', alpha=0.9, s=3, label='State')
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Selected action in 2D')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    return fig