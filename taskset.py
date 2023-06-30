#!/usr/bin/env python

"""
taskset.py - parser for task set from JSON file
"""

import json
import heapq
from collections import defaultdict
from copy import deepcopy
from re import M, S
import sys
import matplotlib.pyplot as plt


class TaskSetJsonKeys(object):
    # Task set
    KEY_TASKSET = "taskset"

    # Task
    KEY_TASK_ID = "taskId"
    KEY_TASK_PERIOD = "period"
    KEY_TASK_WCET = "wcet"
    KEY_TASK_DEADLINE = "deadline"
    KEY_TASK_OFFSET = "offset"
    KEY_TASK_SECTIONS = "sections"

    # Schedule
    KEY_SCHEDULE_START = "startTime"
    KEY_SCHEDULE_END = "endTime"

    # Release times
    KEY_RELEASETIMES = "releaseTimes"
    KEY_RELEASETIMES_JOBRELEASE = "timeInstant"
    KEY_RELEASETIMES_TASKID = "taskId"


class TaskSetIterator:
    def __init__(self, taskSet):
        self.taskSet = taskSet
        self.index = 0
        self.keys = iter(taskSet.tasks)

    def __next__(self):
        key = next(self.keys)
        return self.taskSet.tasks[key]


class TaskSet(object):
    def __init__(self, data):
        self.parseDataToTasks(data)
        self.buildJobReleases(data)

    def parseDataToTasks(self, data):
        taskSet = {}

        for taskData in data[TaskSetJsonKeys.KEY_TASKSET]:
            task = Task(taskData)

            if task.id in taskSet:
                print("Error: duplicate task ID: {0}".format(task.id))
                return

            if task.period < 0 and task.relativeDeadline < 0:
                print("Error: aperiodic task must have positive relative deadline")
                return

            taskSet[task.id] = task

        self.tasks = taskSet

    def buildJobReleases(self, data):
        jobs = []
        self.jobs_time = defaultdict(lambda: [])
        self.job_deadline = defaultdict(lambda: [])

        if TaskSetJsonKeys.KEY_RELEASETIMES in data:  # necessary for sporadic releases
            for jobRelease in data[TaskSetJsonKeys.KEY_RELEASETIMES]:
                releaseTime = jobRelease[TaskSetJsonKeys.KEY_RELEASETIMES_JOBRELEASE]
                taskId = jobRelease[TaskSetJsonKeys.KEY_RELEASETIMES_TASKID]

                job = self.getTaskById(taskId).spawnJob(releaseTime)
                jobs.append(job)
        else:
            self.scheduleStartTime = data[TaskSetJsonKeys.KEY_SCHEDULE_START]
            self.scheduleEndTime = data[TaskSetJsonKeys.KEY_SCHEDULE_END]
            for task in self:
                current_time = max(task.offset, self.scheduleStartTime)
                while current_time < self.scheduleEndTime:
                    job = task.spawnJob(current_time)
                    if job is not None:
                        jobs.append(job)
                    self.jobs_time[current_time].append(job)
                    self.job_deadline[job.deadline].append(job)
                    if task.period >= 0:
                        current_time += task.period  # periodic
                    else:
                        current_time = self.scheduleEndTime  # aperiodic

        self.jobs = jobs

    def __contains__(self, elt):
        return elt in self.tasks

    def __iter__(self):
        return TaskSetIterator(self)

    def __len__(self):
        return len(self.tasks)

    def getTaskById(self, taskId):
        return self.tasks[taskId]

    def printTasks(self):
        print("\nTask Set:")
        for task in self:
            print(task)

    def printJobs(self):
        print("\nJobs:")
        for task in self:
            for job in task.getJobs():
                print(job)

    @property
    def all_sections(self):
        sections = set()
        for task in self:
            for current_time in [i[0] for i in task.sections]:
                sections.add(current_time)
        return sections


class Task(object):
    def __init__(self, taskDict):
        self.id = taskDict[TaskSetJsonKeys.KEY_TASK_ID]
        self.period = taskDict[TaskSetJsonKeys.KEY_TASK_PERIOD]
        self.wcet = taskDict[TaskSetJsonKeys.KEY_TASK_WCET]
        self.relativeDeadline = taskDict.get(TaskSetJsonKeys.KEY_TASK_DEADLINE,
                                             taskDict[TaskSetJsonKeys.KEY_TASK_PERIOD])
        self.offset = taskDict.get(TaskSetJsonKeys.KEY_TASK_OFFSET, 0)
        self.sections = taskDict[TaskSetJsonKeys.KEY_TASK_SECTIONS]

        self.lastJobId = 0
        self.lastReleasedTime = 0
        self.jobs = []

    def getAllResources(self):
        return self.sections

    def spawnJob(self, releaseTime):
        if self.lastReleasedTime > 0 and releaseTime < self.lastReleasedTime:
            print("INVALID: release time of job is not monotonic")
            return None

        if self.lastReleasedTime > 0 and releaseTime < self.lastReleasedTime + self.period:
            print("INVDALID: release times are not separated by period")
            return None

        self.lastJobId += 1
        self.lastReleasedTime = releaseTime

        job = Job(self, self.lastJobId, releaseTime)

        self.jobs.append(job)
        return job

    def getJobs(self):
        return self.jobs

    def getJobById(self, jobId):
        if jobId > self.lastJobId:
            return None

        job = self.jobs[jobId - 1]
        if job.id == jobId:
            return job

        for job in self.jobs:
            if job.id == jobId:
                return job

        return None

    def getUtilization(self):
        return self.wcet / self.period

    def __str__(self):
        return "task {0}: (Φ,T,C,D,∆) = ({1}, {2}, {3}, {4}, {5})".format(self.id, self.offset, self.period, self.wcet,
                                                                          self.relativeDeadline, self.sections)


class Job(object):
    def __init__(self, task: Task, jobId, releaseTime):
        self.task = task
        self.jobId = jobId
        self.releaseTime = releaseTime
        self.is_completed = False
        self.remaining_time = self.task.wcet
        self.sections = deepcopy(self.task.sections)
        self.current_section = 0
        self.deadline = releaseTime + self.task.relativeDeadline
        self.fix_priority = 1/self.task.relativeDeadline
        self.priority = self.fix_priority

    def getResourceHeld(self):
        '''the resources that it's currently holding'''
        return self.sections[self.current_section][0]

    def getRecourseWaiting(self):
        '''a resource that is being waited on, but not currently executing'''
        return self.sections[self.current_section + 1:]

    def getRemainingSectionTime(self):
        return self.sections[self.current_section][1]

    def execute(self):
        remaining = 1
        while remaining > 0:
            time_section = self.getRemainingSectionTime()
            if remaining >= time_section:
                self.sections[self.current_section][1] = 0
                self.current_section += 1
                remaining -= time_section
                if self.current_section >= len(self.sections):
                    self.current_section -= 1
                    self.is_completed = True
                return remaining
            else:
                self.sections[self.current_section][1] -= remaining
                remaining = 0
        return remaining

    def executeToCompletion(self):
        return None

    def isCompleted(self):
        return self.is_completed

    def __str__(self):
        return "[{0}:{1}] released at {2} -> deadline at {3}".format(self.task.id, self.jobId, self.releaseTime,
                                                                     self.deadline)

    def __le__(self, other):
        return 1/self.priority <= 1/other.priority

    def __lt__(self, other):
        return 1/self.priority < 1/other.priority


class NPPScheduler:
    def __init__(self, taskset: TaskSet) -> None:
        self.taskset = taskset
        self.queue = []
        self.future_jobs = []
        self._initial_jobs()

    # Add jobs
    def _initial_jobs(self):
        for task in self.taskset:
            for job in task.getJobs():
                self.future_jobs.append(job)

    def run(self):
        running_job: Job = None
        interval = None
        intervals = []
        current_section = None
        feasibility = True

        # Iteration
        current_time = self.taskset.scheduleStartTime
        while current_time <= self.taskset.scheduleEndTime:

            # Add jobs to queue
            jobs = self.taskset.jobs_time[current_time]
            for job in jobs:
                heapq.heappush(self.queue, job)

            # Run new job
            if len(self.queue) != 0:
                next_job = heapq.heappop(self.queue)

                ## When there is no any running job
                if running_job == None:
                    running_job = next_job

                    ## Start time, End time, isBlocked , is completed , jobId, taskId, resource holded
                    interval = [current_time, current_time, False, False,
                                running_job.jobId, running_job.task.id, running_job.getResourceHeld()]

                # Replace new job with higher priority to running job
                elif running_job.getResourceHeld() == 0 and next_job.priority > running_job.priority:
                    heapq.heappush(self.queue, running_job)
                    if interval[0] != interval[1]:
                        intervals.append(interval)
                    running_job = next_job
                    interval = [current_time, current_time, True, False, running_job.jobId,
                                running_job.task.id, running_job.getResourceHeld()]

                # Running job has a section so we run running job
                elif interval[6] != current_section and current_section != None:
                    heapq.heappush(self.queue, next_job)
                    if interval[0] != interval[1]:
                        intervals.append(interval)
                    heapq.heappush(self.queue, running_job)
                    running_job = heapq.heappop(self.queue)
                    interval = [current_time, current_time, False, False,
                                running_job.jobId, running_job.task.id, running_job.getResourceHeld()]
                else:
                    heapq.heappush(self.queue, next_job)

            # Queue is empty and we have a running job
            elif interval[6] != current_section and current_section != None and running_job != None:
                if interval[0] != interval[1]:
                    intervals.append(interval)
                heapq.heappush(self.queue, running_job)
                running_job = None
                continue

            # We have a job todo
            if running_job != None:

                # Check feasibility
                if running_job.deadline < current_time:
                    feasibility = False
                # Execute
                remaining = running_job.execute()
                interval[1] = interval[1] + 1 - remaining

                # Check completeing job
                if running_job.isCompleted():
                    interval[3] = True
                    if interval[0] != interval[1]:
                        intervals.append(interval)
                    running_job = None
                else:
                    current_section = running_job.getResourceHeld()

                current_time += 1 - remaining
            else:
                current_time += 1
        self.results(intervals, feasibility)

    def results(self, intervals, feasibility):
        for inter in intervals:
            print("interval [{} , {}]: task {}, job {}".format(
                inter[0], inter[1], inter[5], inter[4],))
        if feasibility:
            print("FEASIBLE")
        else:
            print("NOT FEASIBLE")

        task_interval = defaultdict(lambda: [])
        section_color = {0: 'purple', 1: 'coral', 2: 'olive',
                         3: 'khaki', 4: 'cyan', 5: 'gray'}

        for interval in intervals:
            task_interval[interval[5]].append(
                [[interval[0], interval[1] - interval[0]], section_color[interval[6]]])

        fig, ax = plt.subplots()
        ax.set_xlim(-1, self.taskset.scheduleEndTime + 1)

        tmp = 100 // (len(self.taskset.tasks) + 2)
        ticks = [100 - i * tmp for i in range(1, len(self.taskset.tasks) + 1)]
        ax.set_yticks(ticks)
        ax.set_yticklabels(['Task ' + str(i + 1)
                           for i in range(len(self.taskset.tasks))])

        for job in self.future_jobs:
            y = 100 - (job.task.id + 1/2) * tmp
            ax.vlines(x=job.releaseTime, ymin=y, ymax=y + tmp - 2,
                      color="green", linewidth=2, linestyles='dashed')
            ax.vlines(x=job.deadline - 0.5, ymin=y, ymax=y + tmp - 2,
                      color="red", linewidth=2, linestyles='dashed')

        for i, task_id in enumerate(sorted(task_interval.keys())):
            height = 100 - (i + 3/2) * tmp
            ax.broken_barh([a[0] for a in task_interval[task_id]], (height, tmp - 2),
                           facecolors=[a[1] for a in task_interval[task_id]])

        plt.show()


class PIPScheduler:
    def __init__(self, taskset: TaskSet, time_step=1) -> None:
        self.taskset = taskset
        self.queue = []
        self.future_jobs = []
        self._initial_jobs()

    # Add jobs
    def _initial_jobs(self):
        for task in self.taskset:
            for job in task.getJobs():
                self.future_jobs.append(job)

    def run(self):
        running_job: Job = None
        interval = None
        intervals = []
        current_section = None
        feasibility = True
        resources_waiting = dict()

        for section in self.taskset.all_sections:
            resources_waiting[section] = None

        # Iteration
        current_time = self.taskset.scheduleStartTime
        while current_time <= self.taskset.scheduleEndTime:
            jobs = self.taskset.jobs_time[current_time]
            for job in jobs:
                heapq.heappush(self.queue, job)

            if len(self.queue) != 0:
                next_job = heapq.heappop(self.queue)

                if running_job == None:

                    # Set temporarily higher priority to next job
                    if next_job.getResourceHeld() != 0 and resources_waiting[next_job.getResourceHeld()] != None and resources_waiting[next_job.getResourceHeld()] != next_job:
                        self.queue.queue.remove(
                            resources_waiting[next_job.getResourceHeld()])
                        tmp = resources_waiting[next_job.getResourceHeld(
                        )]
                        tmp.priority = next_job.priority
                        heapq.heapify(self.queue.queue)
                        self.queue.push(next_job)
                        next_job = tmp

                    # Add next job's resource to waitings
                    elif next_job.getResourceHeld() != 0:
                        resources_waiting[next_job.getResourceHeld(
                        )] = next_job

                    # Run new job
                    running_job = next_job
                    interval = [current_time, current_time, False, False,
                                running_job.jobId, running_job.task.id, running_job.getResourceHeld()]

                # We have a running job. Now we have to check priority of now and next jobs
                elif next_job.priority > running_job.priority:
                    prev_job = running_job
                    heapq.heappush(self.queue, running_job)

                    # Update priority of next job
                    if next_job.getResourceHeld() != 0 and resources_waiting[next_job.getResourceHeld()] != None:
                        self.queue.queue.remove(
                            resources_waiting[next_job.getResourceHeld()])
                        tmp = resources_waiting[next_job.getResourceHeld(
                        )]
                        tmp.priority = next_job.priority
                        heapq.heapify(self.queue.queue)
                        heapq.heappush(self.queue, next_job)
                        next_job = tmp

                    # Add resource of next job to waiting
                    elif next_job.getResourceHeld() != 0:
                        resources_waiting[next_job.getResourceHeld(
                        )] = next_job

                    # Run job
                    running_job = next_job
                    if running_job != prev_job:
                        if interval[0] != interval[1]:
                            intervals.append(interval)
                        interval = [current_time, current_time, True, False, running_job.jobId,
                                    running_job.task.id, running_job.getResourceHeld()]

                # Run running job
                elif interval[6] != current_section and current_section != None and running_job != None:
                    if interval[0] != interval[1]:
                        intervals.append(interval)

                    running_job.priority = running_job.fix_priority
                    heapq.heappush(self.queue, running_job)
                    heapq.heappush(self.queue, next_job)
                    resources_waiting[interval[6]] = None

                    running_job = None
                    next_job = heapq.heappop(self.queue)
                    if next_job.getResourceHeld() != 0 and resources_waiting[next_job.getResourceHeld()] != None:
                        self.queue.remove(
                            resources_waiting[next_job.getResourceHeld()])
                        tmp = resources_waiting[next_job.getResourceHeld()]
                        tmp.priority = next_job.priority
                        heapq.heapify(self.queue)
                        heapq.heappush(self.queue, next_job)
                        next_job = tmp
                    elif next_job.getResourceHeld() != 0:
                        resources_waiting[next_job.getResourceHeld()
                                          ] = next_job
                    running_job = next_job
                    interval = [current_time, current_time, False, False,
                                running_job.jobId, running_job.task.id, running_job.getResourceHeld()]

                # Add next job to heap
                else:
                    heapq.heappush(self.queue, next_job)

            # Queue is empty and we have a running job
            elif interval[6] != current_section and current_section != None and running_job != None:
                if interval[0] != interval[1]:
                    intervals.append(interval)
                running_job.priority = running_job.fix_priority
                heapq.heappush(self.queue, running_job)

                resources_waiting[interval[6]] = None

                running_job = None
                continue

            # Run the chosen job
            if running_job != None:
                if running_job in self.taskset.job_deadline[current_time]:
                    feasibility = False
                remaining = running_job.execute()
                interval[1] = interval[1] + 1 - remaining

                if running_job.isCompleted():
                    interval[3] = True
                    if interval[0] != interval[1]:
                        intervals.append(interval)
                    running_job.priority = running_job.fix_priority
                    resources_waiting[interval[6]] = None
                    running_job = None
                else:
                    current_section = running_job.getResourceHeld()

                current_time += 1 - remaining
            else:
                current_time += 1
        self.results(intervals, feasibility)

    def results(self, intervals, feasibility):
        for inter in intervals:
            print(
                "interval [{} , {}]: task {}, job {}".format(
                    inter[0],
                    inter[1],
                    inter[5],
                    inter[4],
                )
            )
        if feasibility:
            print("FEASIBLE")
        else:
            print("NOT FEASIBLE")

        task_interval = defaultdict(lambda: [])
        section_color = {0: 'purple', 1: 'coral', 2: 'olive',
                         3: 'khaki', 4: 'cyan', 5: 'gray'}

        for interval in intervals:
            task_interval[interval[5]].append(
                [[interval[0], interval[1] - interval[0]], section_color[interval[6]]])

        fig, ax = plt.subplots()
        ax.set_xlim(-1, self.taskset.scheduleEndTime + 1)

        tmp = 100 // (len(self.taskset.tasks) + 2)
        ticks = [100 - i * tmp for i in range(1, len(self.taskset.tasks) + 1)]
        ax.set_yticks(ticks)
        ax.set_yticklabels(['Task ' + str(i + 1)
                           for i in range(len(self.taskset.tasks))])

        for job in self.future_jobs:
            y = 100 - (job.task.id + 1/2) * tmp
            ax.vlines(x=job.releaseTime, ymin=y, ymax=y + tmp - 2,
                      color="green", linewidth=2, linestyles='dashed')
            ax.vlines(x=job.deadline - 0.5, ymin=y, ymax=y + tmp - 2,
                      color="red", linewidth=2, linestyles='dashed')

        for i, task_id in enumerate(sorted(task_interval.keys())):
            height = 100 - (i + 3/2) * tmp
            ax.broken_barh([a[0] for a in task_interval[task_id]], (height, tmp - 2),
                           facecolors=[a[1] for a in task_interval[task_id]])

        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[0]
        mode = sys.argv[1]
    else:
        file_path = "taskset2.json"

    with open(file_path) as json_data:
        data = json.load(json_data)

    taskSet = TaskSet(data)

    taskSet.printTasks()
    taskSet.printJobs()

    PIPScheduler(taskSet).run()
    # NPPScheduler(taskSet).run()
