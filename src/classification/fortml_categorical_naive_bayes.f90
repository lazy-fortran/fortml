module fortml_categorical_naive_bayes
    !! Weighted Categorical Naive Bayes with deterministic category metadata.
    !!
    !! Features are integer category codes.  Categories are sorted separately
    !! for every feature and stored in one packed vector.  Unknown categories
    !! either raise a domain error (the default) or contribute a neutral
    !! likelihood when `handle_unknown` is enabled.  Input derivatives are
    !! intentionally refused because category lookup is discrete.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    real(dp), parameter :: DEFAULT_ALPHA = 1.0_dp

    type, public :: categorical_naive_bayes_t
        private
        integer, allocatable :: categories(:), offsets(:)
        real(dp), allocatable :: log_probability(:, :), prior(:), weighted_count(:)
        integer, allocatable :: class_label(:)
        integer :: n_features = 0, n_classes = 0
        real(dp) :: alpha = DEFAULT_ALPHA
        logical :: handle_unknown = .false., is_fitted = .false.
    contains
        procedure, public :: fit => categorical_nb_fit
        procedure, public :: predict_log_proba => categorical_nb_predict_log_proba
        procedure, public :: predict_proba => categorical_nb_predict_proba
        procedure, public :: predict => categorical_nb_predict
        procedure, public :: predict_proba_jvp => categorical_nb_predict_proba_jvp
        procedure, public :: classes => categorical_nb_classes
        procedure, public :: category_values => categorical_nb_category_values
        procedure, public :: category_offsets => categorical_nb_category_offsets
        procedure, public :: category_count => categorical_nb_category_count
        procedure, public :: class_prior => categorical_nb_class_prior
        procedure, public :: weighted_class_counts => categorical_nb_weighted_counts
        procedure, public :: alpha_value => categorical_nb_alpha
        procedure, public :: fitted => categorical_nb_fitted
        procedure, public :: parameter_count => categorical_nb_parameter_count
    end type categorical_naive_bayes_t

    public :: categorical_nb_fit

contains

    subroutine categorical_nb_fit(self, x, labels, status, alpha, priors, &
            sample_weight, class_weight, handle_unknown)
        class(categorical_naive_bayes_t), intent(out) :: self
        integer, intent(in) :: x(:, :), labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha, priors(:), sample_weight(:), &
            class_weight(:)
        logical, intent(in), optional :: handle_unknown
        integer, allocatable :: classes(:), unique_values(:), counts(:)
        real(dp), allocatable :: weights(:)
        real(dp) :: requested_alpha, total_weight, prior_sum, count
        integer :: n_samples, n_features, n_classes, total_categories
        integer :: i, j, c, k, first, last

        self%is_fitted = .false.
        requested_alpha = DEFAULT_ALPHA
        if (present(alpha)) requested_alpha = alpha
        if (.not. ieee_is_finite(requested_alpha) .or. requested_alpha <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CategoricalNB fit: alpha must be finite and positive")
            return
        end if
        self%handle_unknown = .false.
        if (present(handle_unknown)) self%handle_unknown = handle_unknown
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1 .or. size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CategoricalNB fit: input dimensions are invalid")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CategoricalNB fit: at least two classes are required")
            return
        end if
        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CategoricalNB fit: sample weights are invalid")
                return
            end if
            weights = sample_weight
        end if
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CategoricalNB fit: class weights are invalid")
                return
            end if
            do i = 1, n_samples
                do c = 1, n_classes
                    if (labels(i) == classes(c)) then
                        weights(i) = weights(i)*class_weight(c)
                        exit
                    end if
                end do
            end do
        end if
        total_weight = sum(weights)
        if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CategoricalNB fit: effective weights must have positive mass")
            return
        end if

        allocate(self%offsets(n_features + 1))
        self%offsets(1) = 1
        total_categories = 0
        do j = 1, n_features
            call sorted_unique_column(x(:, j), unique_values)
            if (size(unique_values) < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CategoricalNB fit: every feature needs a category")
                return
            end if
            total_categories = total_categories + size(unique_values)
            self%offsets(j + 1) = total_categories + 1
        end do
        allocate(self%categories(total_categories), &
            self%log_probability(total_categories, n_classes))
        total_categories = 0
        do j = 1, n_features
            call sorted_unique_column(x(:, j), unique_values)
            first = total_categories + 1
            last = total_categories + size(unique_values)
            self%categories(first:last) = unique_values
            total_categories = last
        end do
        allocate(self%prior(n_classes), self%weighted_count(n_classes), &
            self%class_label(n_classes))
        self%class_label = classes
        self%weighted_count = 0.0_dp
        do c = 1, n_classes
            do i = 1, n_samples
                if (labels(i) == classes(c)) self%weighted_count(c) = &
                    self%weighted_count(c) + weights(i)
            end do
            if (self%weighted_count(c) <= 0.0_dp .or. &
                .not. ieee_is_finite(self%weighted_count(c))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CategoricalNB fit: every class needs positive effective weight")
                return
            end if
        end do
        do j = 1, n_features
            first = self%offsets(j)
            last = self%offsets(j + 1) - 1
            do c = 1, n_classes
                do k = first, last
                    count = 0.0_dp
                    do i = 1, n_samples
                        if (labels(i) == classes(c) .and. x(i, j) == self%categories(k)) &
                            count = count + weights(i)
                    end do
                    self%log_probability(k, c) = log((count + requested_alpha)/ &
                        (self%weighted_count(c) + requested_alpha*real(last-first+1, dp)))
                end do
            end do
        end do
        if (present(priors)) then
            if (size(priors) /= n_classes .or. any(.not. ieee_is_finite(priors)) .or. &
                any(priors <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CategoricalNB fit: priors are invalid")
                return
            end if
            self%prior = priors
        else
            self%prior = self%weighted_count
        end if
        prior_sum = sum(self%prior)
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CategoricalNB fit: prior mass is invalid")
            return
        end if
        self%prior = self%prior/prior_sum
        self%n_features = n_features
        self%n_classes = n_classes
        self%alpha = requested_alpha
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine categorical_nb_fit

    subroutine categorical_nb_predict_log_proba(self, x, log_probabilities, status)
        class(categorical_naive_bayes_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: joint(self%n_classes), maximum, normalizer
        integer :: i, j, c, index

        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(log_probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CategoricalNB log probability: model or shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            joint = log(self%prior)
            do j = 1, self%n_features
                index = category_index(self, j, x(i, j))
                if (index == 0) then
                    if (.not. self%handle_unknown) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "CategoricalNB log probability: unknown category")
                        return
                    end if
                else
                    joint = joint + self%log_probability(index, :)
                end if
            end do
            maximum = maxval(joint)
            normalizer = sum(exp(joint - maximum))
            if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CategoricalNB log probability: normalization is invalid")
                return
            end if
            log_probabilities(i, :) = joint - maximum - log(normalizer)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine categorical_nb_predict_log_proba

    subroutine categorical_nb_predict_proba(self, x, probabilities, status)
        class(categorical_naive_bayes_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_probabilities(size(probabilities, 1), size(probabilities, 2))

        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(log_probabilities)
        call status_set(status, FORTNUM_OK, "")
    end subroutine categorical_nb_predict_proba

    subroutine categorical_nb_predict(self, x, labels, status)
        class(categorical_naive_bayes_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: probabilities(size(x, 1), self%n_classes)
        integer :: i

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CategoricalNB predict: output shape is invalid")
            return
        end if
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine categorical_nb_predict

    subroutine categorical_nb_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(categorical_naive_bayes_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        real(dp), intent(in) :: x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "CategoricalNB input JVP is undefined for discrete category lookup")
    end subroutine categorical_nb_predict_proba_jvp

    function categorical_nb_classes(self) result(values)
        class(categorical_naive_bayes_t), intent(in) :: self
        integer, allocatable :: values(:)
        allocate(values, source=self%class_label)
    end function categorical_nb_classes

    function categorical_nb_category_values(self) result(values)
        class(categorical_naive_bayes_t), intent(in) :: self
        integer, allocatable :: values(:)
        allocate(values, source=self%categories)
    end function categorical_nb_category_values

    function categorical_nb_category_offsets(self) result(values)
        class(categorical_naive_bayes_t), intent(in) :: self
        integer, allocatable :: values(:)
        allocate(values, source=self%offsets)
    end function categorical_nb_category_offsets

    integer function categorical_nb_category_count(self, feature) result(count)
        class(categorical_naive_bayes_t), intent(in) :: self
        integer, intent(in) :: feature
        count = 0
        if (feature >= 1 .and. feature <= self%n_features) then
            count = self%offsets(feature+1) - self%offsets(feature)
        end if
    end function categorical_nb_category_count

    function categorical_nb_class_prior(self) result(values)
        class(categorical_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        allocate(values, source=self%prior)
    end function categorical_nb_class_prior

    function categorical_nb_weighted_counts(self) result(values)
        class(categorical_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        allocate(values, source=self%weighted_count)
    end function categorical_nb_weighted_counts

    real(dp) function categorical_nb_alpha(self) result(value)
        class(categorical_naive_bayes_t), intent(in) :: self
        value = self%alpha
    end function categorical_nb_alpha

    logical function categorical_nb_fitted(self) result(value)
        class(categorical_naive_bayes_t), intent(in) :: self
        value = self%is_fitted
    end function categorical_nb_fitted

    integer function categorical_nb_parameter_count(self) result(count)
        class(categorical_naive_bayes_t), intent(in) :: self
        count = 0
        if (self%is_fitted) count = size(self%log_probability) + self%n_classes
    end function categorical_nb_parameter_count

    integer function category_index(self, feature, value) result(index)
        class(categorical_naive_bayes_t), intent(in) :: self
        integer, intent(in) :: feature, value
        integer :: low, high, middle

        index = 0
        low = self%offsets(feature)
        high = self%offsets(feature+1) - 1
        do while (low <= high)
            middle = (low + high)/2
            if (self%categories(middle) == value) then
                index = middle
                return
            else if (self%categories(middle) < value) then
                low = middle + 1
            else
                high = middle - 1
            end if
        end do
    end function category_index

    subroutine sorted_unique_column(values, unique_values)
        integer, intent(in) :: values(:)
        integer, allocatable, intent(out) :: unique_values(:)
        integer, allocatable :: work(:)
        integer :: i, count

        allocate(work(size(values)))
        work = values
        call sort_in_place(work)
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(i-1)) count = count + 1
        end do
        allocate(unique_values(count))
        unique_values(1) = work(1)
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(i-1)) then
                count = count + 1
                unique_values(count) = work(i)
            end if
        end do
    end subroutine sorted_unique_column

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        call sorted_unique_column(labels, classes)
    end subroutine sorted_unique_labels

    subroutine sort_in_place(values)
        integer, intent(inout) :: values(:)
        integer :: i, j, value
        do i = 2, size(values)
            value = values(i)
            j = i - 1
            do while (j >= 1)
                if (values(j) <= value) exit
                values(j+1) = values(j)
                j = j - 1
            end do
            values(j+1) = value
        end do
    end subroutine sort_in_place

end module fortml_categorical_naive_bayes
