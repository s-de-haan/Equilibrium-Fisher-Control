@torch.no_grad()
    def dynamical_inversion(self, output_target, verbose=True):
        r"""Compute the dynamical (simulated) inversion of the targets.

        It does the inversion in real time, controlling all hidden layers
        simultaneously.

        This function calls ``self.controller()`` as a subroutine, which
        returns values for :math:`\mathbf{u}`, :math:`\mathbf{v}^\text{fb}`,
        :math:`\mathbf{v}`, :math:`\mathbf{v}^\text{ff}` and :math:`\mathbf{r}`
        for every simulated time step. The last values of these arrays are taken
        to represent the steady state. However, convergence is not guaranteed.
        If ``self.include_non_converged_samples`` is set to  ``False``,
        the values of batch elements that did not converge are set to
        their feedforward mode values, which includes :math:`\mathbf{u}` and
        :math:`\mathbf{v}^\text{fb}` being set to 0. 
        If ``self.include_non_converged_samples`` is set to ``True``, some of 
        the returned values with ``_ss`` suffix may in fact not represent the 
        steady state.

        Args:
            output_target (torch.Tensor): The output targets.
            verbose (bool): Whether to display warnings.
        
        Returns:
            (....): An ordered tuple containing:

            - **u_ss** (torch.Tensor): :math:`\mathbf{u}`, the final control
              input.
            - **v_ss** (list):
              :math:`\mathbf{v}_{ss} = \mathbf{v}^- + \
              \Delta_{\mathbf{v}}`
              The final voltage activations of the somatic
              compartments, split in a list that contains
              :math:`\mathbf{v}_{ss}` for each layer.
            - **r_ss** (list): :math:`\mathbf{r}_{ss} = \
              \phi(\mathbf{v}_{ss})`. The final firing rates of the
              neurons, split in a list that contains :math:`\mathbf{r}_{ss}`
              for each layer.
            - **r_out_ss** (torch.Tensor): The finaloutput activation
              of the network.
            - **delta_v_ss** (list): A list containing the final
              :math:`\Delta \mathbf{v}_i` for each layer.
            - **(u, v_fb, v, v_ff, r)** (tuple): A tuple with 5 elements.
              :math:`\mathbf{u}` represents a tensor of dimension
              :math:`t_{max}\times B \times n_L` containing the control
              input for each timestep.
              :math:`\mathbf{v}^\text{fb}`, :math:`\mathbf{v}`,
              :math:`\mathbf{v}^\text{ff}` each contain a list with at
              index ``i`` a ``torch.Tensor`` of dimension
              :math:`t_{max}\times B \times n_i`.
              :math:`\mathbf{r}` is a list with at index ``i`` a
              ``torch.Tensor`` of dimension :math:`t_{max}\times B \times n_i`
              containing the firing rates of layer ``i`` for each timestep.

        """
        batch_size = self.layers[0].activations.shape[0]

        # Get the post- and pre-nonlinearity activations.
        r_feedforward = [l.activations for l in self.layers]
        v_feedforward = [l.linear_activations for l in self.layers]

        # Compute the target activations.
        r, u, (v_fb, v_ff, v), sample_error = \
            self.controller(output_target, self.alpha_di, self.dt_di,
                            self.tmax_di,
                            k_p=self.k_p,
                            noisy_dynamics=self.noisy_dynamics,
                            inst_transmission=self.inst_transmission,
                            time_constant_ratio=self.time_constant_ratio,
                            apical_time_constant=self.apical_time_constant,
                            proactive_controller=self.proactive_controller,
                            sigma=self.sigma,
                            sigma_output=self.sigma_output)

        converged, diverged = self.check_convergence(r, r_feedforward,
                                                     output_target, u, 
                                                     sample_error, batch_size)

        # Select only samples that have converged.
        non_conv_idxs = converged == 0
        non_conv_idxs = mutils.bool_to_indices(non_conv_idxs)
        if not self.include_non_converged_samples:
            for i in range(self.depth):
                v[i][:, non_conv_idxs, :] = v_feedforward[i][non_conv_idxs, :]
                v_ff[i][:, non_conv_idxs, :] = v_feedforward[i][non_conv_idxs,:]
                v_fb[i][:, non_conv_idxs, :] = 0.
                r[i][:, non_conv_idxs, :] = r_feedforward[i][non_conv_idxs]
            u[:, non_conv_idxs, :] = 0.
            if verbose:
                warnings.warn('There are %s non-converged '%len(non_conv_idxs)+\
                              'samples that are discarded.')
        elif len(non_conv_idxs) > 0:
            if verbose:
                warnings.warn('There are %s non-converged '%len(non_conv_idxs)+\
                              'samples in the mini-batch.')

        # Get the steady-state target values (i.e. at last timestep).
        r_ss = [val[-1] for val in r]
        v_fb_ss = [val[-1] for val in v_fb]
        v_ff_ss = [val[-1] for val in v_ff]
        v_ss = [val[-1] for val in v]
        r_out_ss = r_ss[-1]
        u_ss = u[-1]

        # Compute the difference in somatic and basal voltages at steady-state.
        delta_v_ss = [v_ss[i] - v_ff_ss[i] for i in range(len(v_ss))]

        return u_ss, v_ss, r_ss, r_out_ss, delta_v_ss, \
                (u, v_fb, v, v_ff, r)

def controller(self, output_target, alpha, dt, tmax, k_p=0.,
                   noisy_dynamics=False, inst_transmission=False,
                   time_constant_ratio=1., apical_time_constant=-1,
                   proactive_controller=False, sigma=0.01, sigma_output=0.01):
        r"""Simulate the feedback control loop for several timesteps. 

        The following continuous time ODEs are simulated with time interval
        ``dt``. The following equation is used for the voltage:

        .. math::

            \frac{\tau_v}{\tau_u}\frac{d \mathbf{v}_i(t)}{dt} = \
                -\mathbf{v}_i(t) + W_i \mathbf{r}_{i-1}(t) + b_i + \
                Q_i \mathbf{u}(t)
            
        And the following for the control signal:

        .. math::

            \mathbf{u}(t) = \mathbf{u}^{\text{int}}(t) + k \mathbf{e}(t)

        .. math::

            \tau_u \frac{d \mathbf{u}^{\text{int}}(t)}{dt} = \mathbf{e}(t) - \
                \alpha \mathbf{u}^{\text{int}}(t)

        Note that we use a ratio :math:`\frac{\tau_v}{\tau_u}` instead of two
        separate time constants for :math:`\mathbf{v}` and :math:`\mathbf{u}`,
        as a scaling of both time constants can be absorbed in the simulation
        timestep ``dt``.
        IMPORTANT: ``time_constant_ratio`` should never be taken smaller than
        ``dt``, as then the forward Euler method will become unstable by
        default (the simulation steps will start to 'overshoot').

        If ``inst_transmission=False``, the forward Euler method is used to
        simulate the differential equation. If ``inst_transmission=True``, a
        slight modification is made to the forward Euler method, assuming that
        we have instant transmission from one layer to the next: the basal
        voltage of layer ``i`` at timestep ``t`` will already be based on the
        forward propagation of the somatic voltage of layer ``i-1`` at timestep
        ``t``, hence including the feedback of layer ``i-1`` at timestep ``t``.
        It is recommended to put ``inst_transmission=True`` when the
        ``time_constant_ratio`` is approaching ``dt``, as then we are
        approaching the limit of instantaneous system dynamics in the simulation
        where ``inst_transmission`` is always used (See below).

        If ``inst_system_dynamics=True``, we assume that the time constant of
        the system (i.e. the network) is much smaller than that of the
        controller and we approximate this by replacing the dynamical equations
        for :math:`\mathbf{v}_i` by their instantaneous equivalents:

        .. math::

            \mathbf{v}_i(t) = W_i \mathbf{r}_{i-1}(t) + b_i + Q_i \mathbf{u}(t)

        Note that ``inst_transmission`` will always be put on ``True`` 
        (overridden) in combination with ``inst_system_dynamics``.

        If ``proactive_controller=True``, the control input ``u[k+1]`` will be
        used to compute the apical voltages ``v^\text{fb}[k+1]``, instead of the
        control input ``u[k]``. This is a slight variation on the forward Euler
        method and corresponds to the conventional discretized control schemes.

        If ``noisy_dynamics=True``, noise is added to the apical compartment of
        the neurons. We now simulate the apical compartment with its own
        dynamics, as the feedback learning rule needs access to the noisy apical
        compartment. We use the following stochastic differential equation for
        the apical compartment:
        
        .. math::
        
            \tau_{\text{fb}} d \mathbf{v}_i^{\text{fb}}(t) = \
                (-\mathbf{v}_i^{\text{fb}}(t) + Q_i \mathbf{u}(t))dt + \sigma \
                \bm{\epsilon}_i(t)
        
        with :math:`\bm{\epsilon}` the Wiener process (Brownian motion) with
        covariance matrix :math:`I`.

        This is simulated with the Euler-Maruyama method:
        
        .. math::
        
            v_i^\text{fb}[k+1] = v_i^\text{fb}[k] + \Delta t / \tau_\text{fb} \
                (-v_i^\text{fb}[k] + Q_i u[k]) + \sigma / \sqrt{\Delta t / \
                \tau_\text{fb}} \Delta \beta

        with :math:`\Delta \beta` drawn from the zero-mean Gaussian distribution
        with covariance :math:`I`. The other dynamical equations in the system
        remain the same, except that :math:`Q_i \mathbf{u}` is replaced by
        :math:`\mathbf{v}_i^\text{fb}`:

        .. math::
        
            \tau_v \frac{d \mathbf{v}_i(t)}{dt} = -\mathbf{v}_i(t) + W_i \
                \mathbf{r}_{i-1}(t) + b_i + \mathbf{v}_i^\text{fb}

        One can opt for instantaneous apical compartment dynamics by putting
        its time constant :math:`\tau_\text{fb}` (``apical_time_constant``) equal
        to ``dt``. This is not encouraged for training the feedback weights, but
        can be used for simulating noisy system dynamics for training the
        forward weights, resulting in:

        .. math::

            \tau_v d \mathbf{v}_i(t) = (-\mathbf{v}_i(t) + W_i \
                \mathbf{r}_{i-1}(t) + b_i + Q_i \mathbf{u}(t) )dt + \
                \sigma \bm{\epsilon}_i(t)

        which can again be similarly discretized with the Euler-Maruyama method.

        Note that for training the feedback weights, it is recommended to put
        ``inst_transmission=True``, such that the noise of all layers can
        influence the output at the current timestep, instead of having to wait
        for a couple of timesteps, depending on the layer depth.

        Note that in the current implementation, we interpret that the noise is
        added in the apical compartment, and that the basal and somatic
        compartments are not noisy. At some point we might want to also add
        noise in the somatic and basal compartments for physical realism.

        Args:
            output_target (torch.Tensor): The output target
                :math:`\mathbf{r}_L^*` that is used by the controller to compute
                the control error :math:`\mathbf{e}(t)`.
            alpha (float): The leakage term of the controller.
            dt (float): The time interval used in the forward Euler method.
            tmax (int): The maximum number of timesteps.
            k_p (float): The positive gain parameter for the proportional part
                of the controller. If it is equal to zero (by default),
                no proportional control will be used, only integral control.
            noisy_dynamics (bool): Flag indicating whether noise should be
                added to the dynamics.
            inst_transmission (bool): Flag indicating whether the modified
                version of the forward Euler method should be used, where it is
                assumed that there is instant transmission between layers (but
                not necessarily instant voltage dynamics). See the docstring
                above for more information.
            time_constant_ratio (float): Ratio of the time constant of the
                voltage dynamics w.r.t. the controller dynamics.
            apical_time_constant (float): Time constant of the apical
                compartment. If ``-1``, we assume that the user does not want
                to model the apical compartment dynamics, but assumes instant
                transmission to the somatic compartment instead (i.e. apical
                time constant of zero).

        Returns:

            (....): Tuple containing:

            - **r** (list): A list with at index ``i`` a ``torch.Tensor``
                of dimension :math:`t_{max}\times B \times n_i` containing the
                firing rates of layer ``i`` for each timestep.
            - **u** (torch.Tensor): A tensor of dimension
                :math:`t_{max}\times B \times n_L` containing the control input
                for each timestep.
            - **(v_fb, v_ff, v)** (tuple): A tuple with 3 elements, each
                containing a list with at index ``i`` a ``torch.Tensor`` of
                dimension :math:`t_{max}\times B \times n_i` containing the
                voltage levels of the apical, basal or somatic compartments
                respectively.
            - **sample_error** (torch.Tensor): A tensor of dimension
                :math:`t_{max} \times B` containing the L2 norm of the error
                :math:`\mathbf{e}(t)` at each timestep.
        """
        if k_p < 0:
            raise ValueError('Only positive values for "k_p" are allowed')
        if self.inst_system_dynamics:
            inst_transmission = True

        if apical_time_constant == -1 or apical_time_constant == None:
            apical_time_constant = dt
        assert apical_time_constant > 0

        # Extract important variables and shapes.
        batch_size = output_target.shape[0]
        L = len(self.layers) 
        lod = [l.weights.shape[0] for l in self.layers] # layer out dims
        size_output = output_target.shape[1]
        tmax = int(tmax)
        device = output_target.device

        # Create empty containers for desired variables:
        # - v_fb: apical voltage (Ki u)
        # - v_ff: basal voltage (Wi h_target_i-1)
        # - v: somatic voltage
        # - r: v after non-linearlity
        # - u: control signal
        # - u_int: intermediate control signal if proportional part is active
        v_fb = [torch.zeros((tmax, batch_size, l),device=device) for l in lod]
        v_ff = [torch.zeros((tmax, batch_size, l),device=device) for l in lod]
        v = [torch.zeros((tmax, batch_size, l),device=device) for l in lod]
        r = [torch.zeros((tmax, batch_size, l),device=device) for l in lod]
        u = torch.zeros((tmax, batch_size, size_output),device=device)
        if k_p > 0:
            u_int = torch.zeros((tmax, batch_size, size_output),device=device)
        u_lp = None
        v_lp = None
        if self.low_pass_filter_u:
            u_lp = torch.zeros_like(u,device=device)
        if self.low_pass_filter_noise:
            noise_filtered = [torch.zeros((batch_size, l),device=device) for \
                              l in lod]
        if noisy_dynamics and self.use_jacobian_as_fb:
            v_lp = [torch.zeros((tmax, batch_size, l),device=device)\
                   for l in lod]
        sample_error = torch.ones((tmax, batch_size),device=device) * 10
        
        # Fill the values at the initial timestep.
        for i in range(L):
            v_ff[i][0, :] = self.layers[i].linear_activations
            v[i][0, :] = self.layers[i].linear_activations
            r[i][0, :] = self.layers[i].activations    
            if v_lp is not None:
                v_lp[i][0, :] = self.layers[i].linear_activations      
        sample_error[0] = self.compute_loss(output_target, r[-1][0, :])

        # Save initial zero targets for computation of Jacobian if needed.
        self._r = [r_l[:1, :] for r_l in r]

        # If hidden activations are linear, then J doen't depend on the samples
        if self.use_jacobian_as_fb and self.activation == 'linear':
            J = self.compute_full_jacobian(noisy_dynamics=noisy_dynamics)

        # Iterate over all the timesteps.
        for t in range(tmax - 1):

            # Compute the error.
            e = self.compute_error(output_target, r[-1][t])

            # If hidden activations are nonlinear, then J does depend on the
            # samples (derivative of their activations).
            if self.use_jacobian_as_fb and self.activation != 'linear':
                J = self.compute_full_jacobian(noisy_dynamics=noisy_dynamics)
            
            # Compute the control signal ``u``.
            if k_p > 0.:
                # Proportional and integral control.
                u_int[t + 1] = u_int[t] + dt * (e - alpha * u[t])
                u[t + 1] = u_int[t + 1] + k_p * e
            else:
                # Only integral control.
                u[t + 1] = u[t] + dt * (e - alpha * u[t])
            # Exponential low-pass filter if necessary.
            if self.low_pass_filter_u:
                # We need to keep track both of the unfiltered u and the
                # low-pass filtered u, as we might need the high-frequency parts
                # of u for the single-phase feedback weight updates.
                if t == 0:
                    # start the low-pass filtering at the same value of u,
                    # as otherwise it takes a long time to recover from zero
                    u_lp[t + 1] = u[t + 1]
                else:
                    u_lp[t + 1] = (dt / self.tau_f) * u[t + 1] + \
                                  (1 - (dt / self.tau_f)) * u_lp[t]

            def layer_iteration(i):
                """Compute the controlled activations of layer ``i``."""
                # Get the activities of previous layer.
                if i == 0:
                    r_previous = self.input
                else:
                    if inst_transmission:
                        r_previous = r[i - 1][t + 1]
                    else:
                        r_previous = r[i - 1][t]

                # Get basal voltage of current layer (based on ff input).
                a = r_previous.mm(self.layers[i].weights.t())
                if self.layers[i].bias is not None:
                    a += self.layers[i].bias.unsqueeze(0).expand_as(a)
                v_ff[i][t + 1, :] = a

                def get_control_signal(t, u_aux):
                    """Get the control signal Qu for the given timestep.

                    By default, this computes :math:`Qu` but in case the option
                    `use_jacobian_as_fb``is active, this computes :math:`Ju`.

                    Args:
                        t (int): The timestep.
                        u_aux (torch.Tensor): The control u to use. Can be
                            low-pass filtered or not, depending on
                            `low_pass_filter_u`.

                    Returns:
                        (torch.Tensor): The control signal.
                    """

                    if self.use_jacobian_as_fb:
                        batch_size = u_aux.shape[1]
                        n_out = u_aux.shape[2]

                        # Select the correct Jacobian block.
                        J_sq = J.view(batch_size * n_out, J.shape[-1])
                        Ji = mutils.split_in_layers(self, J_sq)[i]
                        Ji = Ji.view(batch_size, n_out, Ji.shape[-1])

                        return torch.matmul(u_aux[t].unsqueeze(1), Ji).squeeze()
                    else:
                        return torch.mm(u_aux[t], \
                                        self.layers[i].weights_backward.t())

                # Get the control signal.
                control_signal = get_control_signal(\
                                    t + 1 if proactive_controller else t,
                                    u_lp if self.low_pass_filter_u else u)
                assert control_signal.shape == v_fb[i][t, :].shape

                # Get apical voltage of current layer (based on fb input).
                if self.inst_apical_dynamics:
                    v_fb[i][t + 1, :] = control_signal
                else:
                    v_fb[i][t + 1, :] = v_fb[i][t, :] + dt / apical_time_constant *\
                                      (- v_fb[i][t, :] + control_signal)

                # Add noise to the apical voltage if necessary.
                if noisy_dynamics:
                    sigma_copy = sigma
                    if i == self.depth - 1:
                        sigma_copy = sigma_output
                    if self.low_pass_filter_noise:
                        # Warning: for very small dt, we might need to change
                        # the implementation for numerical stability and work
                        # with tau_noise * sqrt(dt) instead of 
                        # alpha_noise / sqrt(dt).
                        alpha_noise = dt / self.tau_noise
                        noise_filtered[i] = \
                                (alpha_noise / np.sqrt(dt)) * \
                                torch.randn_like(v_fb[i][t + 1, :],\
                                device=device)+\
                                (1 - alpha_noise) * noise_filtered[i]
                        v_fb[i][t + 1, :] +=  sigma_copy * noise_filtered[i]
                    else:   
                        v_fb[i][t + 1, :] +=  sigma_copy * np.sqrt(dt) / \
                            apical_time_constant * \
                            torch.randn_like(v_fb[i][t + 1, :],device=device)

                # Get somatic voltage as function of basal and apical voltages.
                if self.inst_system_dynamics:
                    v[i][t + 1, :] = v_fb[i][t + 1, :] + v_ff[i][t + 1, :]
                else: 
                    v[i][t + 1, :] = v[i][t, :] + dt / time_constant_ratio \
                                      * (v_fb[i][t + 1, :] + v_ff[i][t + 1, :] -
                                         v[i][t, :])

                # Compute the post-nonlinearity activations of the layer.
                r[i][t + 1, :] = \
                    self.layers[i].forward_activation_function(v[i][t + 1, :])

                # Update activations in layer objects to enable steady-state 
                # jacobian computation in `compute_full_jacobian()`
                if self.use_jacobian_as_fb:
                    if self.activation != 'linear':
                        self.layers[i].linear_activations = v[i][t + 1, :]
                        self.layers[i].activations = r[i][t + 1, :]
                    if noisy_dynamics:
                        alpha_r = dt / self.tau_f
                        v_lp[i][t + 1, :] = alpha_r * self.v[i][t + 1] + \
                                            (1 - alpha_r) * v_lp[i][t]
                        self.layers[i].linear_activations_lp = \
                                        [v_lp_l[:t + 1, :] for v_lp_l in v_lp]

            # Computed the controlled activations of all layers in current ts.
            if not inst_transmission:
                # Compute backwards to have already the influence of the
                # controller being propagated through network across time
                for i in range(L - 1, 0 - 1, -1):
                    layer_iteration(i)
            else:
                for i in range(L):
                    layer_iteration(i)

            # Compute the loss.
            sample_error[t + 1] = self.compute_loss(output_target,
                                                    r[-1][t + 1, :])

            # Save targets for computation of Jacobian if needed, only up to
            # the current timestep.
            self._r = [r_l[:t + 1, :] for r_l in r]

        # Store the control signal.
        if noisy_dynamics:
            # With noisy dynamics, the last value of u will be noisy, and we
            # should average over u to cancel out the noise. I assume that in
            # the last quarter of the simulation, u has converged, so we can
            # average over that interval.
            interval_length = int(tmax / 4)
            self.u = torch.sum(u[-interval_length:-1,:,:], dim=0)\
                    /float(interval_length)
        else:
            self.u = u[-1]

        return r, u, (v_fb, v_ff, v), sample_error
