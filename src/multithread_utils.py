from queue import Queue
from threading import Thread, Lock

class Worker(Thread):
    """
    Thread executing tasks from a given tasks queue. 
    """
    def __init__(self, tasks, thread_id, logger=None):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.id = thread_id
        self.logger = logger
        self.start()

    def run(self):
        while True:
            # extract arguments and organize them properly
            func, args, kargs = self.tasks.get()
            if self.logger :
                self.logger.debug("[Thread %d] Args retrieved: \"%s\"" % (self.id, args))
            new_args = []
            if self.logger :
                self.logger.debug("[Thread %d] Length of args: %d" % (self.id, len(args)))
            for a in args[0]:
                new_args.append(a)
            new_args.append(self.id)
            if self.logger :
                self.logger.debug("[Thread %d] Length of new_args: %d" % (self.id, len(new_args)))
            try:
                # call the function with the arguments previously extracted
                func(*new_args, **kargs)
            except Exception as e:
                # an exception happened in this thread
                if self.logger :
                    self.logger.error(traceback.format_exc())
                else :
                    print(traceback.format_exc())
            finally:
                # mark this task as done, whether an exception happened or not
                if self.logger :
                    self.logger.debug("[Thread %d] Task completed." % self.id)
                self.tasks.task_done()

        return

class ThreadPool:
    """
    Pool of threads consuming tasks from a queue.
    """
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for i in range(num_threads):
            Worker(self.tasks, i)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))
        return

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)
        return

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()
        return

