@torch.no_grad()
    def _dynamical_inversion(self):
        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]

        v_fb_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_ff_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        u_current = torch.zeros((self.bzs, self.output_size))
        u_int_current = torch.zeros((self.bzs, self.output_size))

        for i, layer in enumerate(self.layers):
            v_ff_current[i] = layer.linear_activations
            v_current[i] = layer.linear_activations
            r_current[i] = layer.activations
        
        # Controller loop # TODO: while u(t) not converged
        for _ in range(self.tmax - 1):
            _, Jis = self._calculate_full_jacobian()
            error = self.targets - r_current[-1]

            # Proportional and integral (PI) control.
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                r_previous = r_current[i - 1] if i != 0 else self.input

                # Forward
                a = r_previous.mm(layer.weights.t())
                a += layer.bias.unsqueeze(0).expand_as(a)
                v_ff_current[i] = a

                # Apical input (Ju)
                v_fb_current[i] = torch.matmul(u_next.unsqueeze(1), Jis[i]).squeeze()

                # Soma with apical
                v_current[i] = v_current[i] + (self.dt / self.time_constant_ratio) * (
                    v_fb_current[i] + v_ff_current[i] - v_current[i]
                )
                r_current[i] = layer.activation_fn(v_current[i])
                layer.linear_activations = v_current[i]
                layer.activations = r_current[i]

            u_int_current = u_int_next
            u_current = u_next

        # Steady-state values per layer
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])