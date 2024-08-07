from __future__ import annotations

from collections import defaultdict
from columnflow.tasks.ml import MLEvaluation
from columnflow.tasks.reduction import MergeReducedEventsUser, MergeReducedEvents
from columnflow.tasks.framework.base import Requirements, DatasetTask, AnalysisTask, wrapper_factory
from columnflow.tasks.framework.remote import RemoteWorkflow

from columnflow.tasks.framework.mixins import (
    CalibratorsMixin,
    SelectorMixin,
    ProducersMixin,
    MLModelMixin,
    ChunkedIOMixin,
)
from columnflow.util import dev_sandbox, maybe_import
import law

np = maybe_import("numpy")
ak = maybe_import("awkward")



class MLEvaluationPerFold(MLEvaluation):

    # @law.dynamic_workflow_condition
    # def workflow_condition(self):
    #     # declare that the branch map can be built if the workflow requirement exists
    #     # note: self.input() refers to the outputs of tasks defined in workflow_requires()
    #     return self.input()["events"].exists()

    additional_metrics = law.OptionalBoolParameter(
        description="create additional metrics, e.g. shap values (default: false)",
        default=False,
        significant=False,
    )

    # @workflow_condition.create_branch_map
    def create_branch_map(self):

        overarching_branch_map = super().create_branch_map()
        branches = [
            {
                "parquet": k,
                "fold": f
            }
            for k in range(len(overarching_branch_map))
            for f in range(self.ml_model_inst.folds)
        ]
        # from IPython import embed; embed()
        return {
            i: entry for i, entry in enumerate(branches)
        }

    def workflow_requires(self):
        reqs = super().workflow_requires()
        # from IPython import embed; embed()

        reqs["models"] = self.reqs.MLTraining.req_different_branching(
            self,
            configs=(self.config_inst.name,),
            calibrators=(self.calibrators,),
            selectors=(self.selector,),
            producers=(self.producers,),
        )

        reqs["events"] = self.reqs.MergeReducedEvents.req_different_branching(
            self,
            tree_index=-1,
            branch=-1,
            _exclude={"branches"}
        )

        # add producer dependent requirements
        if self.preparation_producer_inst:
            reqs["preparation_producer"] = self.preparation_producer_inst.run_requires()

        if not self.pilot and self.producer_insts:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req_different_branching(
                    self,
                    producer=producer_inst.cls_name
                )
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]

        return reqs

    def requires(self):
        branch_data = self.branch_map[self.branch]
        reqs = super().requires()
        reqs.update(
            {
                "models": self.reqs.MLTraining.req_different_branching(
                    self,
                    configs=(self.config_inst.name,),
                    calibrators=(self.calibrators,),
                    selectors=(self.selector,),
                    producers=(self.producers,),
                    branch=-1,
                ),
                "events": self.reqs.MergeReducedEvents.req_different_branching(
                    self,
                    tree_index=branch_data["parquet"],
                    # branch=branch_data["parquet"][0],
                    branch=0,
                    # _exclude={"branch", "tree_index"},
                ),
            }
        )

        if self.preparation_producer_inst:
            reqs["preparation_producer"] = self.preparation_producer_inst.run_requires()

        if self.producer_insts:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req_different_branching(
                    self,
                    producer=producer_inst.cls_name,
                    # branch=branch_data["parquet"][0],
                    branch=branch_data["parquet"],
                )
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]

        return reqs

    # @workflow_condition.output
    @MergeReducedEventsUser.maybe_dummy
    def output(self):
        k = self.ml_model_inst.folds
        branch_data = self.branch_map[self.branch]
        return {"mlcolumns": self.target(f"mlevents_{branch_data['parquet']}_f{branch_data['fold']}of{k}.parquet")}

    @law.decorator.log
    @law.decorator.localize
    @law.decorator.safe_output
    def run(self):
        from columnflow.columnar_util import (
            Route, RouteFilter, sorted_ak_to_parquet, update_ak_array, add_ak_aliases,
        )

        # prepare inputs and outputs
        reqs = self.requires()
        inputs = self.input()
        output = self.output()
        output_chunks = {}
        stats = defaultdict(float)

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # run the setup of the optional producer
        reader_targets = {}
        if self.preparation_producer_inst:
            reader_targets = self.preparation_producer_inst.run_setup(
                reqs["preparation_producer"],
                inputs["preparation_producer"],
            )

        # open all model files
        branch_data = self.branch_map[self.branch]
        current_fold = branch_data["fold"]
        models = self.ml_model_inst.open_model(inputs["models"]["collection"].targets[current_fold])


        # get shift dependent aliases
        aliases = self.local_shift_inst.x("column_aliases", {})

        # check once if the events were used during trainig
        events_used_in_training = self.events_used_in_training(
            self.config_inst,
            self.dataset_inst,
            self.global_shift_inst,
        )

        # define columns that need to be read
        read_columns = {Route("deterministic_seed")}
        read_columns |= set(map(Route, aliases.values()))
        read_columns |= set.union(*self.ml_model_inst.used_columns.values())
        if self.preparation_producer_inst:
            read_columns |= self.preparation_producer_inst.used_columns

        # define columns that will be written
        write_columns = set.union(*self.ml_model_inst.produced_columns.values())
        route_filter = RouteFilter(write_columns)

        # iterate over chunks of events and columns
        files = [inputs["events"]["events"]]
        if self.producer_insts:
            files.extend([inp["columns"] for inp in inputs["producers"]])
        if reader_targets:
            files.extend(reader_targets.values())

        # prepare inputs for localization
        # from IPython import embed; embed()
        with law.localize_file_targets(
            [*files, *reader_targets.values()],
            mode="r",
        ) as inps:
            for (events, *columns), pos in self.iter_chunked_io(
                [inp.path for inp in inps],
                source_type=len(files) * ["awkward_parquet"] + [None] * len(reader_targets),
                read_columns=(len(files) + len(reader_targets)) * [read_columns],
            ):
                # optional check for overlapping inputs
                if self.check_overlapping_inputs:
                    self.raise_if_overlapping([events] + list(columns))

                # add additional columns
                events = update_ak_array(events, *columns)

                # add aliases
                events = add_ak_aliases(
                    events,
                    aliases,
                    remove_src=True,
                    missing_strategy=self.missing_column_alias_strategy,
                )

                # generate fold indices
                fold_indices = events.deterministic_seed % self.ml_model_inst.folds

                # invoke the optional producer
                if len(events) and self.preparation_producer_inst:
                    events = self.preparation_producer_inst(
                        events,
                        stats=stats,
                        fold_indices=fold_indices,
                        ml_model_inst=self.ml_model_inst,
                    )

                # evaluate the model
                events = self.ml_model_inst.evaluate(
                    self,
                    events,
                    models,
                    fold_indices,
                    events_used_in_training=events_used_in_training,
                    fixed_fold_index=current_fold,
                    create_additional_metrics=self.additional_metrics,
                )

                # remove columns
                events = route_filter(events)

                # optional check for finite values
                if self.check_finite_output:
                    self.raise_if_not_finite(events)

                # save as parquet via a thread in the same pool
                chunk = tmp_dir.child(f"file_{pos.index}.parquet", type="f")
                output_chunks[pos.index] = chunk
                self.chunked_io.queue(sorted_ak_to_parquet, (events, chunk.path))

        # merge output files
        sorted_chunks = [output_chunks[key] for key in sorted(output_chunks)]
        law.pyarrow.merge_parquet_task(
            self, sorted_chunks, output["mlcolumns"], local=True, writer_opts=self.get_parquet_writer_opts(),
        )

MLEvaluationPerFoldWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=MLEvaluationPerFold,
    enable=["configs", "skip_configs", "shifts", "skip_shifts", "datasets", "skip_datasets"],
)


class MergeMLEvaluationPerFold(
    MLModelMixin,
    ProducersMixin,
    SelectorMixin,
    CalibratorsMixin,
    ChunkedIOMixin,
    MergeReducedEventsUser,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))
    reqs = Requirements(
        MergeReducedEventsUser.reqs,
        RemoteWorkflow.reqs,
        MLEvaluationPerFold=MLEvaluationPerFold,
        MergeReducedEvents=MergeReducedEvents,
    )

    # strategy for handling missing source columns when adding aliases on event chunks
    missing_column_alias_strategy = "original"

    def workflow_requires(self):
        reqs = super().workflow_requires()
        # from IPython import embed; embed()
        reqs.update(
            {
                "ml_evaluation": self.reqs.MLEvaluationPerFold.req_different_branching(
                    self,
                    branch=-1,
                ),
                "events": self.reqs.MergeReducedEvents.req(self),
            }
        )

        return reqs

    def requires(self):
        reqs = super().requires() or dict()
        reqs.update(
            {
                "ml_evaluation": [
                    self.reqs.MLEvaluationPerFold.req_different_branching(
                        self,
                        branch=self.branch + fold,
                    )
                    for fold in range(self.ml_model_inst.folds)
                ],
                "events": self.reqs.MergeReducedEvents.req(self),
            }
        )

        return reqs

    # @workflow_condition.output
    @MergeReducedEventsUser.maybe_dummy
    def output(self):
        return {"mlcolumns": self.target(f"mlevents_{self.branch}.parquet")}


    @law.decorator.log
    @law.decorator.localize
    @law.decorator.safe_output
    def run(self):
        from columnflow.columnar_util import (
            Route, RouteFilter, sorted_ak_to_parquet, update_ak_array, add_ak_aliases,
            has_ak_column, get_ak_routes, EMPTY_FLOAT, set_ak_column
        )

        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()
        output_chunks = {}

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # get shift dependent aliases
        aliases = self.local_shift_inst.x("column_aliases", {})

        # define columns that need to be read
        read_columns = {Route("deterministic_seed")}
        read_columns |= set(map(Route, aliases.values()))

        # define columns that will be written
        read_columns |= set.union(*self.ml_model_inst.produced_columns.values())
        write_columns = set.union(*self.ml_model_inst.produced_columns.values())
        route_filter = RouteFilter(write_columns)

        # iterate over chunks of events and columns
        files = [inputs["events"].targets[0]["events"]]
        if self.producer_insts:
            files.extend([inp["mlcolumns"] for inp in inputs["ml_evaluation"]])

        # # prepare inputs for localization
        # # from IPython import embed; embed()
        with law.localize_file_targets(
            files,
            mode="r",
        ) as inps:
            for (events, *columns), pos in self.iter_chunked_io(
                [inp.path for inp in inps],
                source_type=len(files) * ["awkward_parquet"],
                read_columns=len(files)* [read_columns],
            ):
                # optional check for overlapping inputs
                if self.check_overlapping_inputs:
                    self.raise_if_overlapping([events] + list(columns))

                # add aliases
                events = add_ak_aliases(
                    events,
                    aliases,
                    remove_src=True,
                    missing_strategy=self.missing_column_alias_strategy,
                )

                # # generate fold indices
                fold_indices = events.deterministic_seed % self.ml_model_inst.folds
                # from IPython import embed; embed()
                for i, arr in enumerate(columns):
                    index_mask = fold_indices == i
                    for route in get_ak_routes(arr):
                        fallback = (
                            route.apply(events)
                            if has_ak_column(events, route)
                            else EMPTY_FLOAT
                        )
                        try:
                            tmp = ak.where(index_mask, route.apply(arr), fallback)
                        except:
                            tmp = ak.where(index_mask[..., None], route.apply(arr), fallback)
                        events = set_ak_column(events, route, tmp)

                # remove columns
                events = route_filter(events)

                # optional check for finite values
                if self.check_finite_output:
                    self.raise_if_not_finite(events)

                # save as parquet via a thread in the same pool
                chunk = tmp_dir.child(f"file_{pos.index}.parquet", type="f")
                output_chunks[pos.index] = chunk
                self.chunked_io.queue(sorted_ak_to_parquet, (events, chunk.path))

        # merge output files
        sorted_chunks = [output_chunks[key] for key in sorted(output_chunks)]
        law.pyarrow.merge_parquet_task(
            self, sorted_chunks, output["mlcolumns"], local=True, writer_opts=self.get_parquet_writer_opts(),
        )


MergeMLEvaluationPerFoldWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=MergeMLEvaluationPerFold,
    enable=["configs", "skip_configs", "shifts", "skip_shifts", "datasets", "skip_datasets"],
)
