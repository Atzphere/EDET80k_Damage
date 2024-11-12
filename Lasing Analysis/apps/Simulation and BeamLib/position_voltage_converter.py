import numpy as np
import os
os.chdir(os.path.dirname(__file__))

PARAMETER_FILE_PATH = r"./calibratedPVCParameters - Copy.txt"

SQUARE_X_OFFSET = 0
SQUARE_Y_OFFSET = 0

X_CHANGE_RATE = 1
Y_CHANGE_RATE = 1

LENS_X_AXIS_OFFSET = -1.5
LENS_Y_AXIS_OFFSET = 16.83

ANNEALING_REGION_HEIGHT = 151

METAL_RING_HEIGHT = 132
METAL_RING_RADIUS = 19.8

WINDOW_THICKNESS = 12
WINDOW_BOTTOM_HEIGHT = 77
WINDOW_TOP_HEIGHT = 89


class ROI(object):
    def __init__(self, side_length=32):
        self.side_length = side_length
 # 32 as of end of june 2024
    def get_side_length_millimeters(self):
        return self.side_length


class PositionVoltageConverter(object):
    def __init__(self):
        """
        Connect this object to the master process and all of its fields. Additionally, initializes fields based off of
        product specifications or approximate measurements, to be subsequently tweaked in calibration.

        :param master_process: the master process calling this
        """
        # the following parameters are updated
        # by the user via the GUI.
        self._region_of_interest = ROI()

        # Default parameter values
        self._base_x_offset = LENS_X_AXIS_OFFSET
        self._base_y_offset = LENS_Y_AXIS_OFFSET
        self._x_mirror_angle_offset = np.pi / 180 * 43
        self._y_mirror_angle_offset = np.pi / 180 * 48.5
        self._x_mirror_z_angle = np.pi / 180 * 15
        self._coordinate_angle_offset = np.pi / 180 * (10)
        # This tilts the entire converter. 
        self._n_window = 1.51
        self._lens_top_height = 65.5
        self._lens_bottom_height = 60
        self._lens_thickness = 5.5
        self._n_lens = 1.88599
        self._lens_lower_rad = 66.7
        self._lens_upper_rad = 259.4
        self._mirror_axis_separation = 16.83

        try:
            with open(PARAMETER_FILE_PATH, 'r') as f:
                params = eval(f.read())
        except FileNotFoundError as e:
            print(e)
            print("not using calibration")
            self._is_calibrated = False
        else:
            self.update_voltage_parameters(params)
            self._is_calibrated = True

        # Position calculator parameters
        self._laser_position = [205, 0, 0]
        self._laser_direction = [-1, 0, 0]

    def update_voltage_parameters(self, new_parameters):
        self._x_mirror_angle_offset = new_parameters.get("X mirror angle offset")
        self._y_mirror_angle_offset = new_parameters.get("Y mirror angle offset")
        self._x_mirror_z_angle = new_parameters.get("X mirror Z angle")
        self._coordinate_angle_offset = new_parameters.get("Coordinate angle offset")
        self._base_x_offset = new_parameters.get("Base X offset")
        self._base_y_offset = new_parameters.get("Base Y offset")

        self._n_window = new_parameters.get("Window refraction index")
        self._lens_top_height = new_parameters.get("Lens top height")
        self._lens_bottom_height = new_parameters.get("Lens bottom height")
        self._lens_thickness = new_parameters.get("Lens thickness")
        self._n_lens = new_parameters.get("Lens refraction index")
        self._lens_lower_rad = new_parameters.get("Lens lower radius of curvature")
        self._lens_upper_rad = new_parameters.get("Lens upper radius of curvature")

    def voltage_given_position(self, position: tuple) -> tuple:
        """
        Given a position, computes the corresponding voltage required to move the laser there.

        The equations found in this method were derived by forming an optical equation for the position of the laser
        given the angles of the two mirrors, and then inverting said equation as far as was analytically possible.
        Also note that due to the time-consuming nature of sympy, the partial derivatives have been pre-calculated.
        For the sake of readability, most paramaters are aliased in various shorthands.

        The following approximations were made in this computation to make a solution possible:
            - Paraxial beam travel through the lenses and window was assumed to permit the use of ray transfer matrices
            - A first degree Taylor approximation was made for the final computation, because the function as it stood
              could not be made explicit with respect to the angles
            - The direction and starting point of the laser can be fine-tuned sufficiently that the y and z components
              of its beam path and origin may be treated as 0.


        :param position: a tuple of mm position coordinates for a point in the region of interest
        :return: a pair of voltages that would move the laser to that position
        """

        position = np.array([position[0] + SQUARE_X_OFFSET - self._region_of_interest.get_side_length_millimeters() / 2,
                             position[1] + SQUARE_Y_OFFSET - self._region_of_interest.get_side_length_millimeters() / 2])

        a_0 = self._x_mirror_angle_offset
        b_0 = self._y_mirror_angle_offset
        theta_0 = self._coordinate_angle_offset
        inverse_rotation_matrix = np.array([[np.cos(theta_0), -1*np.sin(theta_0)],
                                            [np.sin(theta_0), np.cos(theta_0)]])

        base_offsets = np.array([self._base_x_offset, self._base_y_offset])

        c = self._x_mirror_z_angle
        k_helper = (self._n_window - 1) * WINDOW_THICKNESS - self._lens_top_height + ANNEALING_REGION_HEIGHT

        k_1 = (1 + self._lens_thickness * (1 - self._n_lens) / (self._lens_lower_rad * self._n_lens) +
               k_helper * ((self._n_lens - 1)/self._lens_upper_rad+
                           self._lens_thickness * (self._n_lens - 1) * (1 - self._lens_upper_rad) /
                           (self._lens_lower_rad * self._lens_upper_rad * self._n_lens)))

        k_2 = (self._lens_thickness / 2 +
               k_helper * (self._lens_thickness*(self._n_lens - 1)/(self._lens_upper_rad*self._n_lens) + 1))

        r_h_s = (np.dot(inverse_rotation_matrix, position) + base_offsets)

        r = self._mirror_axis_separation
        l_b = self._lens_bottom_height

        x_approx = (k_1*r*(2*np.sin(a_0)**2 - 1)*np.cos(b_0)/(np.sin(2*a_0)*np.cos(b_0 - c)) +
                      (2*np.sin(a_0)**2 - 1)*(l_b + k_2 + r*np.sin(c)*np.cos(b_0)/np.cos(b_0 - c)))

        y_approx = (k_1*r*np.cos(b_0)*np.cos(c)/np.cos(b_0 - c) -
                      (l_b + k_2 + r*np.sin(c)*np.cos(b_0)/np.cos(b_0 - c))*np.sin(2*a_0)*np.cos(2*b_0 - c))

        x_a_partial = 2*(l_b*np.sin(2*a_0 - b_0 + c)/2 + l_b*np.sin(2*a_0 + b_0 - c)/2 + k_1*r*np.cos(b_0)/np.sin(2*a_0)**2 +
                         k_2*np.sin(2*a_0 - b_0 + c)/2 + k_2*np.sin(2*a_0 + b_0 - c)/2 + r*np.cos(-2*a_0 + b_0 + c)/4 -
                         r*np.cos(2*a_0 - b_0 + c)/4 + r*np.cos(2*a_0 + b_0 - c)/4 - r*np.cos(2*a_0 + b_0 + c)/4)/np.cos(b_0 - c)

        x_b_partial = r*(k_1 + 2*np.sin(a_0)*np.sin(c)*np.cos(a_0))*np.sin(c)/(np.cos(b_0 - c)**2*np.tan(2*a_0))

        y_a_partial = -2*(r*np.sin(c)*np.cos(b_0)+(l_b+k_2)*np.cos(b_0 - c))*np.cos(2*a_0)*np.cos(2*b_0 - c)/np.cos(b_0 - c)

        y_b_partial = (k_1*r*np.sin(b_0 - c)*np.cos(b_0)*np.cos(c) + r*np.sin(2*a_0)*np.sin(c)**2*np.cos(2*b_0 - c) +
                       (-k_1*r*np.sin(b_0)*np.cos(c) + 2*(r*np.sin(c)*np.cos(b_0) +
                        (k_2 + l_b)*np.cos(b_0 - c))*np.sin(2*a_0)*np.sin(2*b_0 - c))*np.cos(b_0 - c))/np.cos(b_0 - c)**2

        y_angle = ((r_h_s[1] + y_a_partial*a_0 + y_b_partial * b_0 -
                   y_approx - y_a_partial / x_a_partial * (r_h_s[0] + x_a_partial*a_0 + x_b_partial * b_0 - x_approx))
                   / (y_b_partial - y_a_partial * x_b_partial / x_a_partial))

        x_angle = (r_h_s[0] + x_a_partial * a_0 + x_b_partial * b_0 - x_approx - x_b_partial * y_angle) / x_a_partial

        x_voltage = float((x_angle - self._x_mirror_angle_offset) * 180 / np.pi / 2)
        y_voltage = float((y_angle - self._y_mirror_angle_offset) * 180 / np.pi / 2)

        return x_voltage, y_voltage

    def get_voltage_parameters(self) -> dict:
        return {"X mirror angle offset": self._x_mirror_angle_offset,
                "Y mirror angle offset": self._y_mirror_angle_offset,
                "X mirror Z angle": self._x_mirror_z_angle,
                "Coordinate angle offset": self._coordinate_angle_offset,
                "Base X offset": self._base_x_offset,
                "Base Y offset": self._base_y_offset,
                "Window refraction index": self._n_window,
                "Lens top height": self._lens_top_height,
                "Lens bottom height": self._lens_bottom_height,
                "Lens thickness": self._lens_thickness,
                "Lens refraction index": self._n_lens,
                "Lens lower radius of curvature": self._lens_lower_rad,
                "Lens upper radius of curvature": self._lens_upper_rad}

    def position_given_voltage(self, voltages: tuple, account_for_outer_ring: bool) -> tuple or list[bool]:

        """
        Given a pair of voltages, return the theoretical position of the laser on the ROI plane. ROI must be configured.

        This method is primarily for use in the calibration of this calculator, although it is also used to indicated
        the theoretical position of the laser in the ROI. Note that in the context of the mirrors that are used
        in the annealing environment, the voltages and corresponding angles follow the
        relation [angleX, angleY] = 2 * [voltageX, voltageY].
        The output of this function is dependent on how the ROI has been defined (based off of the square placed in
        the GUI). The region of interest should be approximately centered above the optical axis.

        :param voltages: pair of floats, [x_volt, y_volt], map to a pair of angle positions taken by the mirrors.
        :param account_for_outer_ring: boolean, stipulates whether to treat the lenses as having an outer ring.
        :return: a pair of floats representing a position in the region of interest, [x_mm, y_mm], or [False] in the
                 event that the calculated position is mechanically impossible.
        """
        # TODO find permissible angle range

        return _compute_position(
            voltages, self._laser_position, self._laser_direction, self._base_x_offset,
            self._base_y_offset, self._x_mirror_angle_offset, self._y_mirror_angle_offset,
            self._coordinate_angle_offset, account_for_outer_ring,
            self._region_of_interest.get_side_length_millimeters())

    def is_calibrated(self):
        """
        Determines whether this PositionVoltageConverter is has calibrated parameters

        :return: True if using calibrated parameters, False if relying on default values
        """
        return self._is_calibrated


def _compute_position(voltages: tuple, origin_of_incident_ray, incident_ray_direction, base_x_offset, base_y_offset,
                      x_mirror_angle_offset, y_mirror_angle_offset, coordinate_angle_offset,
                      account_for_outer_ring, square_width) -> tuple or list[bool]:
    """
    This is the main computation function for the theoretical position of the laser beam given mirror angles.

    A broad overview of the computation process is as follows:
    incident laser -> x-mirror reflection -> y-mirror reflection -> refraction from air to lens
    -> refraction from bottom to top of lens -> refraction from lens to vacuum -> refraction through window
    -> coordinate system conversion
    A more detailed description of the optics is available in the documentation.

    General Notes:
    lens : presently the lens is an AC-254-100B, the 3 radii listed refer to the radii of the 3 spheres of curvature,
           and the three "lens" centers refer to the theoretical centers of these spheres.

    window : the window is made of NBK-7

    Coordinate System: Most of the computations are carried out in a coordinate system with its origin on the x mirror.
                       The positive x-axis points towards the origin of the laser, the y points to the right of the
                       beam path, intersecting the rotation axis of the y mirror, and the z axis points to the ceiling.

    Notes on parameters:

    mirror_axis_separation is the distance between mirror axes along a line on the xy plane. It also corresponds to the
    y distance to the centre of the lenses

    """
    # Annealing Setup Optical Parameters
    # values are listed in mm or radians
    annealing_region_height = 151
    metal_ring_height = 132
    metal_ring_radius = 19.8
    x_mirror_z_angle = np.pi / 180 * 15
    mirror_axis_separation = 16.83
    origin = [0, 0, 0]
    lens_centre_1 = np.array([base_x_offset, base_y_offset, 125.8])
    lens_centre_2 = np.array([base_x_offset, base_y_offset, 17])
    lens_centre_3 = np.array([base_x_offset, base_y_offset, -205.6])
    lens_radius_1 = 65.8
    lens_radius_2 = 56
    lens_radius_3 = 280.6
    window_bottom_height = 77
    window_top_height = 89
    n_air = 1
    n_lens_bottom = 1.643
    n_lens_top = 1.783
    n_window = 1.51
    n_vacuum = 1
    lens_casing_radius = 20

    # Begin computation, initialize vectors
    angles = [voltage * 2 for voltage in voltages]

    x_mirror_angle = np.pi / 180 * angles[0] + x_mirror_angle_offset
    y_mirror_angle = np.pi / 180 * angles[1] + y_mirror_angle_offset

    point_on_y_mirror = np.array([0, mirror_axis_separation, 0])
    point_on_x_mirror = np.array(origin)

    x_mirror_normal = np.array([np.sin(x_mirror_angle), np.cos(x_mirror_angle) * np.cos(x_mirror_z_angle),
                                -np.cos(x_mirror_angle) * np.sin(x_mirror_z_angle)])
    y_mirror_normal = np.array([0, np.cos(y_mirror_angle), -np.sin(y_mirror_angle)])

    initial_beam_path = np.array(incident_ray_direction) / np.linalg.norm(np.array(incident_ray_direction))
    beam_origin_point = np.array(origin_of_incident_ray)

    # Compute mirror reflection
    x_reflected_beam = -2 * (np.dot(x_mirror_normal, initial_beam_path)) * x_mirror_normal + initial_beam_path

    x_mirror_intersect_param = np.dot(x_mirror_normal, (point_on_x_mirror - beam_origin_point)) / np.dot(
        x_mirror_normal,
        initial_beam_path)
    x_mirror_intersect_point = beam_origin_point + x_mirror_intersect_param * initial_beam_path
    # TODO: figure out what up with the mirror axis separation

    y_mirror_intersect_param = np.dot(y_mirror_normal, (point_on_y_mirror - x_mirror_intersect_point)) / np.dot(
        y_mirror_normal, x_reflected_beam)

    y_mirror_intersect_point = x_mirror_intersect_point + y_mirror_intersect_param * x_reflected_beam
    y_reflected_beam = -2 * np.dot(y_mirror_normal, x_reflected_beam) * y_mirror_normal + x_reflected_beam

    # Compute refraction through lens
    lens_1_refraction = compute_lens_refraction(lens_centre_1, y_reflected_beam, y_mirror_intersect_point,
                                                lens_radius_1, False, n_air, n_lens_bottom)
    if account_for_outer_ring and not _is_lens_pos_valid(lens_casing_radius, lens_centre_1, lens_1_refraction[1]):
        return [False]

    lens_2_refraction = compute_lens_refraction(lens_centre_2, lens_1_refraction[0], lens_1_refraction[1],
                                                 lens_radius_2, True, n_lens_bottom,
                                                n_lens_top)
    if account_for_outer_ring and not _is_lens_pos_valid(lens_casing_radius, lens_centre_2, lens_2_refraction[1]):
        return [False]

    lens_3_refraction = compute_lens_refraction(lens_centre_3, lens_2_refraction[0], lens_2_refraction[1],
                                                 lens_radius_3, True, n_lens_top,
                                                 n_vacuum)
    if account_for_outer_ring and not _is_lens_pos_valid(lens_casing_radius, lens_centre_3, lens_3_refraction[1]):
        return [False]

    # Compute the refraction through the window
    lens_3_outgoing_point = lens_3_refraction[1]
    lens_3_out_beam = lens_3_refraction[0]

    x_window_bot = np.asscalar(
        lens_3_outgoing_point[0] + (window_bottom_height - lens_3_outgoing_point[2]) / lens_3_out_beam[2] *
        lens_3_out_beam[0])
    y_window_bot = np.asscalar(
        lens_3_outgoing_point[1] + (window_bottom_height - lens_3_outgoing_point[2]) / lens_3_out_beam[2] *
        lens_3_out_beam[1])

    window_bot_intersect = np.array([x_window_bot, y_window_bot, window_bottom_height])

    window_normal = np.array([0, 0, -1])
    bottom_window_beam = n_air / n_window * np.cross(window_normal, np.cross(lens_3_out_beam, window_normal)) - \
                         window_normal * np.sqrt(
        1 - np.square(n_air / n_window) * (1 - np.square(np.dot(window_normal, lens_3_out_beam))))

    x_window_top = np.asscalar(
        window_bot_intersect[0] + (window_top_height - window_bot_intersect[2]) / bottom_window_beam[2] *
        bottom_window_beam[0])
    y_window_top = np.asscalar(
        window_bot_intersect[1] + (window_top_height - window_bot_intersect[2]) / bottom_window_beam[2] *
        bottom_window_beam[1])
    window_out_point = np.array([x_window_top, y_window_top, window_top_height])

    window_out_beam = n_window / n_vacuum * np.cross(window_normal, np.cross(bottom_window_beam, window_normal)) - \
                      window_normal * np.sqrt(
        1 - np.square(n_window / n_vacuum) * (1 - np.square(np.dot(window_normal, bottom_window_beam))))

    # Verify that the theoretical laser position is mechanically possible
    x_outer_ring = np.asscalar(
        window_out_point[0] + (metal_ring_height - window_out_point[2]) / window_out_beam[2] * window_out_beam[
            0])
    y_outer_ring = np.asscalar(
        window_out_point[1] + (metal_ring_height - window_out_point[2]) / window_out_beam[2] * window_out_beam[
            1])
    metal_output_position = np.array([x_outer_ring, y_outer_ring, metal_ring_height])
    if account_for_outer_ring and not _is_lens_pos_valid(metal_ring_radius, lens_centre_3, metal_output_position):
        return [False]

    # Compute that the final position in the initial coordinate system
    x_final = np.asscalar(
        window_out_point[0] + (annealing_region_height - window_out_point[2]) / window_out_beam[2] * window_out_beam[0])
    y_final = np.asscalar(
        window_out_point[1] + (annealing_region_height - window_out_point[2]) / window_out_beam[2] * window_out_beam[1])

    # Transform the position into the final coordinate system
    x_original_coordinates = x_final - base_x_offset
    y_original_coordinates = y_final - base_y_offset
    x_new_coordinates = x_original_coordinates * np.cos(coordinate_angle_offset) + y_original_coordinates * np.sin(
        coordinate_angle_offset)
    y_new_coordinates = -1 * x_original_coordinates * np.sin(coordinate_angle_offset) + y_original_coordinates * np.cos(
        coordinate_angle_offset)
    return [x_new_coordinates - SQUARE_X_OFFSET + square_width / 2,
            y_new_coordinates - SQUARE_Y_OFFSET + square_width / 2]


def compute_lens_refraction(lens_center_point, incident_beam, incoming_point, lens_radius, root_is_positive, n1, n2):
    """
    Helper function to work through the arduous computation of Snell's Law in 3-dimensional vector form.

    Further Reading: https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form

    :param lens_center_point: center of lens (vector)
    :param incident_beam: incident path of laser (vector)
    :param incoming_point: a point along the path of the incident beam (vector)
    :param lens_radius: radius of lens (float)
    :param root_is_positive: whether this computation should involve a positive root (boolean)
    :param n1: refraction index of initial medium (float)
    :param n2: refraction index of final medium (float)
    :return: a list containing the refracted beam and its point of origin on the lens, in that order
    """
    if root_is_positive:
        distance_to_lens = -1 * (np.dot(incident_beam, (incoming_point - lens_center_point))) + np.sqrt(
            np.square(np.dot(incident_beam, (incoming_point - lens_center_point))) - (
                    np.square(np.linalg.norm(incoming_point - lens_center_point)) - lens_radius * lens_radius))
    else:
        distance_to_lens = -1 * (np.dot(incident_beam, (incoming_point - lens_center_point))) - np.sqrt(
            np.square(np.dot(incident_beam, (incoming_point - lens_center_point))) - (
                    np.square(np.linalg.norm(incoming_point - lens_center_point)) - lens_radius * lens_radius))

    surface_intersect = incoming_point + distance_to_lens * incident_beam

    if root_is_positive:
        lens_normal = -1 * (surface_intersect - lens_center_point) / np.linalg.norm(
            surface_intersect - lens_center_point)
    else:
        lens_normal = (surface_intersect - lens_center_point) / np.linalg.norm(surface_intersect - lens_center_point)

    refracted_beam = n1 / n2 * np.cross(lens_normal, np.cross(incident_beam, lens_normal)) - lens_normal * np.sqrt(
        1 - np.square(n1 / n2) * (1 - np.square(np.dot(lens_normal, incident_beam))))

    return [refracted_beam, surface_intersect]


def _is_lens_pos_valid(lens_actual_radius, lens_center_point, surface_intersect):
    """
    Verify that the output position computed by the refraction function is physically possible.

    :param lens_actual_radius: radius of the surface
    :param lens_center_point: center of the surface
    :param surface_intersect: alleged point of intersection with surface
    :return: True if output is possible, false otherwise
    """
    isnan = np.isnan(surface_intersect)
    if isnan[0] or isnan[1] or isnan[2]:
        return False
    elif ((lens_center_point[0] - surface_intersect[0]) ** 2 + (
            lens_center_point[1] - surface_intersect[1]) ** 2) ** 0.5 >= lens_actual_radius:
        return False
    else:
        return True


annealer = PositionVoltageConverter()


def voltage_from_position(x, y):
    return annealer.voltage_given_position((x, y))

def position_from_voltage(xv, yv, **kwargs):
    return annealer.position_given_voltage((xv, yv), **kwargs)