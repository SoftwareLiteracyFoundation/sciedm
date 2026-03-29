from numpy import any, array, concatenate, zeros


def CreateIndices(self):
    """Populate array index vectors lib_i, pred_i
    Indices specified in list of pairs [ 1,10, 31,40... ]
    where each pair is start:stop span of data rows.
    """
    # lib_i from lib
    # libPairs vector of start, stop index pairs
    if len(self._lib) % 2:
        # Odd number of lib
        msg = (
            f"{self._name}: CreateIndices() lib must be an even "
            + "number of elements. Lib start : stop pairs"
        )
        raise RuntimeError(msg)

    libPairs = []  # List of 2-tuples of lib indices
    for i in range(0, len(self._lib), 2):
        libPairs.append((self._lib[i], self._lib[i + 1]))

    # Validate end > start
    for libPair in libPairs:
        libStart, libEnd = libPair

        if self._name in ["Simplex", "SMap", "Multiview"]:
            # Don't check if CCM since default of "1 1" is used.
            if libStart >= libEnd:
                msg = (
                    f"{self._name}: CreateIndices() lib start "
                    + f" {libStart} exceeds lib end {libEnd}."
                )
                raise RuntimeError(msg)

        # Disallow indices < 1, the user may have specified 0 start
        if libStart < 1 or libEnd < 1:
            msg = (
                f"{self._name}: CreateIndices() lib indices "
                + " less than 1 not allowed."
            )
            raise RuntimeError(msg)

    # Loop over each lib pair
    # Add rows for library segments, disallowing vectors
    # in disjoint library gap accommodating embedding and Tp
    embedShift = int(abs(self.tau) * (self._E - 1))
    lib_i_list = list()

    for r in range(len(libPairs)):
        start, stop = libPairs[r]

        # Adjust start, stop to enforce disjoint library gaps
        if not self._embedded:
            if self.tau < 0:
                start = start + embedShift
            else:
                stop = stop - embedShift

        if self.Tp < 0:
            if not self._embedded:
                start = int(max(start, start + abs(self.Tp)) - 1)
        else:
            if r == len(libPairs) - 1:
                stop = stop - int(self.Tp)

        libPair_i = [i - 1 for i in range(start, stop + 1)]

        lib_i_list.append(array(libPair_i, dtype=int))

    # Concatenate lib_i_list into lib_i
    self.lib_i_ = concatenate(lib_i_list)

    if len(lib_i_list) > 1:
        self._disjointLib = True

    # Validate lib_i: E, tau, Tp combination
    if self._name in ["Simplex", "SMap", "CCM", "Multiview"]:
        if self._embedded:
            if len(self.lib_i_) < abs(self.Tp):
                msg = (
                    f"{self._name}: CreateIndices(): embbeded True "
                    + f"Tp = {self.Tp} is invalid for the library."
                )
                raise RuntimeError(msg)
        else:
            vectorStart = max([-embedShift, 0, self.Tp])
            vectorEnd = min([-embedShift, 0, self.Tp])
            vectorLength = abs(vectorStart - vectorEnd) + 1

            if vectorLength > len(self.lib_i_):
                msg = (
                    f"{self._name}: CreateIndices(): Combination of E = "
                    + f"{self._E}  Tp = {self.Tp}  tau = {self.tau} "
                    + "is invalid for the library."
                )
                raise RuntimeError(msg)

    # pred_i from pred
    # predPairs vector of start, stop index pairs
    if len(self._pred) % 2:
        # Odd number of pred
        msg = (
            f"{self._name}: CreateIndices() pred must be an even "
            + "number of elements. Pred start : stop pairs"
        )
        raise RuntimeError(msg)

    predPairs = []  # List of 2-tuples of pred indices
    for i in range(0, len(self._pred), 2):
        predPairs.append((self._pred[i], self._pred[i + 1]))

    if len(predPairs) > 1:
        self.disjointPred = True

    # Validate end > start
    for predPair in predPairs:
        predStart, predEnd = predPair

        if self._name in ["Simplex", "SMap", "Multiview"]:
            # Don't check CCM since default of "1 1" is used.
            if predStart >= predEnd:
                msg = (
                    f"{self._name}: CreateIndices() pred start "
                    + f" {predStart} exceeds pred end {predEnd}."
                )
                raise RuntimeError(msg)

        # Disallow indices < 1, the user may have specified 0 start
        if predStart < 1 or predEnd < 1:
            msg = (
                f"{self._name}: CreateIndices() pred indices "
                + " less than 1 not allowed."
            )
            raise RuntimeError(msg)

    # Create pred_i indices from predPairs
    for r in range(len(predPairs)):
        start, stop = predPairs[r]
        pred_i = zeros(stop - start + 1, dtype=int)

        i = 0
        for j in range(start, stop + 1):
            pred_i[i] = j - 1  # apply zero-offset
            i = i + 1

        self._predList.append(pred_i)  # Append disjoint segment(s)

    # flatten arrays in self._predList for single array self.pred_i_
    pred_i_ = []
    for pred_i in self._predList:
        i_ = [i for i in pred_i]
        pred_i_ = pred_i_ + i_

    self.pred_i_ = array(pred_i_, dtype=int)

    self.PredictionValid()

    self._pred_i_all = self.pred_i_.copy()  # Before nan are removed

    # Remove embedShift nan from predPairs
    # NOTE : This does NOT redefine self.pred_i_, only self.predPairs
    #        self.pred_i_ is redefined to remove all nan in RemoveNan()
    #        at the API level.
    if not self._embedded:
        # If [0, 1, ... embedShift] nan (negative tau) or
        # [N - embedShift, ... N-1, N]  (positive tau) nan
        # are in pred_i delete elements
        nan_i_start = [i for i in range(embedShift)]
        nan_i_end = [self._Data.shape[0] - 1 - i for i in range(embedShift)]

        for i in range(len(self._predList)):
            pred_i = self._predList[i]

            if self.tau > 0:
                if any([i in nan_i_end for i in pred_i]):
                    pred_i_ = [i for i in pred_i if i not in nan_i_end]
                    self._predList[i] = array(pred_i_, dtype=int)
            else:
                if any([i in nan_i_start for i in pred_i]):
                    pred_i_ = [i for i in pred_i if i not in nan_i_start]
                    self._predList[i] = array(pred_i_, dtype=int)

    # Validate lib_i pred_i do not exceed data
    if self.lib_i_[-1] >= self._Data.shape[0]:
        msg = (
            f"{self._name}: CreateIndices() The prediction index "
            + f"{self.lib_i_[-1]} exceeds the number of data rows "
            + f"{self._Data.shape[0]}"
        )
        raise RuntimeError(msg)

    if self.pred_i_[-1] >= self._Data.shape[0]:
        msg = (
            f"{self._name}: CreateIndices() The prediction index "
            + f"{self.pred_i_[-1]} exceeds the number of data rows "
            + f"{self._Data.shape[0]}"
        )
        raise RuntimeError(msg)

    # Check for lib : pred overlap for knn leave-one-out
    if len(set(self.lib_i_).intersection(set(self.pred_i_))):
        self._libOverlap = True

    if self._name == "SMap":
        if self._knn < 1:  # default knn = 0, set knn value to full lib
            self._knn = len(self.lib_i_) - 1

            # if self.verbose:
            #    msg = (
            #        f"{self._name} CreateIndices(): "
            #        + f"Set knn = {self._knn} for SMap."
            #    )
            #    print(msg, flush=True)
