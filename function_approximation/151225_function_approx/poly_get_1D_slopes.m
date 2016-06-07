function slopes = poly_get_1D_slopes(x,y,tolerance)
assert(numel(x) == numel(y));
slopes = [];
previous_slope = [];
for i = 1:(numel(x)-1)
    current_slope = (y(i+1) - y(i))./(x(i+1) - x(i));
    if isempty(previous_slope)
        slopes = [slopes current_slope];
    else
        relative_change = abs(current_slope - previous_slope)./abs(previous_slope);
        if relative_change > tolerance
            slopes = [slopes current_slope];
        end
    end
    previous_slope = current_slope;
end
